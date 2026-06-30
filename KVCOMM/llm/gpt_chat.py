"""Chat backends and local HF model with KV reuse and anchor-based prefill.

This module provides two implementations:
- GPTChat: a thin adapter over an OpenAI-compatible chat API.
- LLMChat: a Hugging Face model runner that supports KV reuse between
  agents/requests and dense prefill with anchor selection.

Key concepts:
- Prefix KV: KV cache segments corresponding to static prompt parts.
- Placeholder: Markers in the prompt whose content is populated per request.
- Anchor: A remembered KV delta used to adjust prefix/payload caches.
"""
from typing import List, Union, Optional, Dict, Any, Tuple
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import threading
import asyncio
import async_timeout
from openai import AsyncOpenAI
import re
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from transformers.cache_utils import DynamicCache
from KVCOMM.llm.format import Message
from KVCOMM.llm.llm import LLM
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.config import KVCommConfig

from KVCOMM.llm.token_ops import *
from KVCOMM.llm.kvcomm_engine import KVCOMMEngine, _RequestState
from KVCOMM.utils.metrics import GenerationResult
from KVCOMM.utils.log import logger


def _short_message_key(message: str, *, limit: int = 72) -> str:
    text = str(message or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _log_input_anchor_routing(
    *,
    node_id: str,
    message_key: str,
    placeholder_id: str,
    reuse_kind: str,
    routing_mode: str,
) -> None:
    logger.debug(
        "[kvcomm-reuse] node={} message_key={} placeholder={} reuse_kind={} input_routing={}",
        node_id,
        _short_message_key(message_key),
        placeholder_id,
        reuse_kind,
        routing_mode,
    )

MINE_API_KEYS = os.getenv('API_KEY')


def _escape_loguru_markup(text: Optional[str]) -> str:
    """Escape Loguru markup tokens in free-form text."""
    if text is None:
        return ""
    return text.replace("<", "\\<")


def _trim_token_ids_at_eos(tokenizer: Any, token_ids: torch.Tensor) -> torch.Tensor:
    """Trim generated token ids at the first EOS to drop HF padding tail."""
    if token_ids is None or token_ids.numel() == 0:
        return token_ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return token_ids
    if token_ids.dim() > 1:
        row = token_ids[0]
        eos_hits = (row == eos_id).nonzero(as_tuple=False)
        if eos_hits.numel() == 0:
            return token_ids
        end = int(eos_hits[0].item()) + 1
        return token_ids[:, :end]
    eos_hits = (token_ids == eos_id).nonzero(as_tuple=False)
    if eos_hits.numel() == 0:
        return token_ids
    end = int(eos_hits[0].item()) + 1
    return token_ids[:end]


_CHAT_TEMPLATE_LEAK_RE = re.compile(
    r"<\|im_start\|>\s*|<\|im_end\|>\s*|<\|redacted_im_end\|>\s*",
    re.IGNORECASE,
)


def _sanitize_chat_template_leaks(text: str) -> str:
    if not text:
        return text
    cleaned = _CHAT_TEMPLATE_LEAK_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_LATENCY_IO_LOCK = threading.Lock()


def _resolve_latency_path(target: Optional[Union[str, Path]]) -> Optional[Path]:
    if target is None:
        return None
    path = Path(target)
    if path.suffix:
        return path
    return path / "Latency.json"


def _append_latency_record(target: Optional[Union[str, Path]], record: Dict[str, Any]) -> None:
    """Persist a latency record to JSON, tolerating malformed or missing files."""
    path = _resolve_latency_path(target)
    if path is None:
        return
    serializable = {key: value for key, value in record.items() if value is not None}
    with _LATENCY_IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        existing = loaded
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(serializable)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)


class _TTFTTracer(StoppingCriteria):
    """Stopping criteria to capture time-to-first-token during generation."""

    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft: Optional[float] = None

    def reset(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if self.ttft is None and input_ids.shape[-1] > self.prompt_length:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.ttft = perf_counter() - self.start_time
        return False


class _TokenStreamCallback(StoppingCriteria):
    """Emit decoded token deltas during generation for OpenClaw SSE streaming."""

    def __init__(self, tokenizer: Any, prompt_length: int, on_token: Any):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.on_token = on_token
        self._last_len = prompt_length

    def reset(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self._last_len = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids is None or input_ids.numel() == 0:
            return False
        # With past_key_values, HF often passes only the newest token each step.
        if input_ids.shape[-1] == 1:
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        else:
            cur_len = int(input_ids.shape[-1])
            if cur_len <= self._last_len:
                return False
            new_ids = input_ids[0, self._last_len:cur_len]
            self._last_len = cur_len
            if new_ids.numel() == 0:
                return False
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        if text:
            self.on_token(text)
        return False


def _bench_no_think_enabled() -> bool:
    raw = os.environ.get("KVCOMM_BENCH_NO_THINK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_MAX_PREFIX_LENGTH_DRIFT = int(os.environ.get("KVCOMM_PREFIX_LENGTH_DRIFT", "16") or "16")
_TURN_PLACEHOLDER_RE = re.compile(r"^turn_\d+_(assistant|tool)$")
_AGENT_CURRENT_PLACEHOLDER_RE = re.compile(r"^agent_(\d+)_current$")


def _reconcile_prefix_kv_and_tokens(
    merged_prefix_kv: Any,
    merged_prefix_token_ids: Dict[str, Any],
) -> Tuple[int, int]:
    """Return (kv_prefix_length, token_input_length), tolerating small KV/token drift."""
    kv_length = int(merged_prefix_kv.get_seq_length())
    input_length = int(merged_prefix_token_ids["input_ids"].shape[-1])
    if input_length == kv_length:
        return kv_length, input_length
    drift = input_length - kv_length
    abs_drift = abs(drift)
    soft_cap = _MAX_PREFIX_LENGTH_DRIFT
    hard_cap = max(soft_cap, int(max(kv_length, input_length) * 0.02), 64)
    if abs_drift > hard_cap:
        logger.error(
            "prefix_token_length: {} merged_length: {} (drift {} exceeds hard cap {})",
            kv_length,
            input_length,
            drift,
            hard_cap,
        )
        raise RuntimeError("prefix_token_length != merged_prefix_token_ids['input_ids'].shape[-1]")
    if abs_drift > soft_cap:
        logger.warning(
            "prefix KV/token drift kv={} tokens={} (drift={} > soft {}); using token_ids for generation bounds",
            kv_length,
            input_length,
            drift,
            soft_cap,
        )
    else:
        logger.warning(
            "prefix KV/token drift kv={} tokens={} (drift={}); using token_ids for generation bounds",
            kv_length,
            input_length,
            drift,
        )
    return kv_length, input_length


def _chat_template_kwargs() -> Dict[str, Any]:
    if not _bench_no_think_enabled():
        return {}
    return {"enable_thinking": False}

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    """Call an OpenAI-compatible chat endpoint asynchronously."""
    api_kwargs = dict(api_key = MINE_API_KEYS)
    try:
        aclient = AsyncOpenAI(**api_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create the async client: {e}")
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    """Thin wrapper around OpenAI-style chat completions."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        """Asynchronously generate a response via hosted chat API."""

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        response_text = await achat(self.model_name, messages)
        metadata: Dict[str, Any] = {}
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        return GenerationResult(
            text=response_text,
            mode="default",
            ttft=0.0,
            metadata=metadata,
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation not implemented for this adapter."""
        pass

@LLMRegistry.register('LLMChat')
class LLMChat(LLM):
    """Local HF model chat with KV reuse and anchor-based dense prefill.

    Provides utilities to construct chat prompts, manage shared KV caches
    across agents/requests, and generate with two strategies:
    - kv_reuse: reuses previously materialized KV segments
    - dense_prefill: regenerates prefix KV and optionally sets anchors
    """
    _shared_model = None
    _shared_tokenizer = None
    _model_lock = threading.Lock()                                        
    _THREAD_POOL: ThreadPoolExecutor | None = None
    _THREAD_POOL_WORKERS: int | None = None
    _shared_kv_cache_memory = None
    _initialization = {}
    anchors = KVCOMMEngine.anchors
    anchor_dict = KVCOMMEngine.anchor_dict
    anchor_len_dict = KVCOMMEngine.anchor_len_dict
    anchor_info_dict = KVCOMMEngine.anchor_info_dict
    weight_dict = KVCOMMEngine.weight_dict
    global_anchor_info_dict = KVCOMMEngine.global_anchor_info_dict

    _request_lock = KVCOMMEngine._request_lock
    _request_states = KVCOMMEngine._request_states
    _active_requests = KVCOMMEngine._active_requests
    _staged_commits = KVCOMMEngine._staged_commits

    def __init__(self, model_name: str, prefix: str = None, config: KVCommConfig | None = None):
        """Create a chat model instance and initialize shared resources.

        Args:
            model_name: HF model identifier.
            prefix: Optional legacy/template prefix configuration.
            config: KVComm runtime configuration.
        """
        self.model_name = model_name

        self.config = (config or KVCommConfig.from_env()).validate()
        self._ensure_thread_pool(self.config.thread_pool_workers)
        self.kv_engine = KVCOMMEngine(self)

        self.lock = asyncio.Lock()                       


        self._initialize_shared_resources()


        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._shared_kv_cache_memory = LLMChat._shared_kv_cache_memory
        self._initialization = LLMChat._initialization
        self._chat_markers = self._extract_chat_markers()
        self.default_assistant_prompt = "A: "
        self.base_messages_template: List[Dict[str, str]] = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        if prefix is not None:
            self._prepare_prefix_template(prefix)

    def _extract_chat_markers(self) -> Dict[str, str]:
        """Parse tokenizer chat template to identify structural markers."""
        template = getattr(self.tokenizer, "chat_template", "") or ""
        markers = {"begin": "", "start": "", "end": "", "eot": ""}
        begin_candidates = ["<|begin_of_text|>", "<s>", getattr(self.tokenizer, "bos_token", "") or ""]
        start_candidates = ["<|start_header_id|>", "<|im_start|>"]
        end_candidates = ["<|end_header_id|>", "<|im_end|>", "\n"]
        eot_candidates = ["<|eot_id|>", "<|im_end|>", getattr(self.tokenizer, "eos_token", "") or ""]

        for token in begin_candidates:
            if token and token in template:
                markers["begin"] = token
                break
        if not markers["begin"]:
            markers["begin"] = begin_candidates[-1]

        for token in start_candidates:
            if token and token in template:
                markers["start"] = token
                break

        for token in end_candidates:
            if token and token in template:
                markers["end"] = token
                break
        if not markers["end"]:
            markers["end"] = ""

        for token in eot_candidates:
            if token and token in template:
                markers["eot"] = token
                break
        if not markers["eot"]:
            markers["eot"] = eot_candidates[-1]

        return markers

    def _prepare_prefix_template(self, prefix: Union[str, List[Dict[str, str]]]) -> None:
        """Normalise various prefix formats into a base messages template."""
        if isinstance(prefix, list):
            self.base_messages_template = prefix
            return
        if isinstance(prefix, dict):
            self.base_messages_template = [prefix]
            return
        if isinstance(prefix, tuple):
            prefix = list(prefix)
        if isinstance(prefix, list) and all(isinstance(item, tuple) and len(item) == 2 for item in prefix):
            self.base_messages_template = [{"role": role, "content": tmpl} for role, tmpl in prefix]
            return
        if isinstance(prefix, str):
            self.default_assistant_prompt = self._extract_assistant_prompt(prefix)
            return
        raise TypeError("Unsupported prefix template type.")

    def _extract_assistant_prompt(self, legacy_prefix: str) -> str:
        """Extract trailing assistant prompt from a legacy text prefix."""
        start = self.start_header_id
        end = self.end_header_id
        if start and end:
            marker = f"{start}assistant{end}\n"
            if marker in legacy_prefix:
                tail = legacy_prefix.split(marker, 1)[-1]
                eot = self.eot_id
                if eot:
                    tail = tail.replace(eot, "")
                return tail
        return legacy_prefix

    @property
    def begin_of_text(self) -> str:
        return self._chat_markers.get("begin", "")

    @property
    def start_header_id(self) -> str:
        return self._chat_markers.get("start", "")

    @property
    def end_header_id(self) -> str:
        return self._chat_markers.get("end", "")

    @property
    def eot_id(self) -> str:
        return self._chat_markers.get("eot", "")

    @staticmethod
    def _normalise_messages(messages: Union[List[Message], List[Dict[str, str]], Dict[str, Any], Tuple[Any, ...], str]) -> List[Dict[str, str]]:
        """Convert mixed message representations into chat dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, tuple):
            if len(messages) == 2 and all(isinstance(item, str) for item in messages):
                system_prompt, user_prompt = messages
                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            return LLMChat._normalise_messages(list(messages))
        if isinstance(messages, dict):
            result: List[Dict[str, str]] = []
            system_prompt = messages.get("system") or messages.get("system_prompt")
            if system_prompt:
                result.append({"role": "system", "content": system_prompt})
            conversation = messages.get("messages") or messages.get("conversation")
            if conversation is not None:
                result.extend(LLMChat._normalise_messages(conversation))
            else:
                if "user" in messages:
                    user_payload = messages["user"]
                    if isinstance(user_payload, list):
                        result.extend(LLMChat._normalise_messages(user_payload))
                    else:
                        result.append({"role": "user", "content": user_payload})
                if "assistant" in messages:
                    assistant_payload = messages["assistant"]
                    if isinstance(assistant_payload, list):
                        result.extend(LLMChat._normalise_messages(assistant_payload))
                    else:
                        result.append({"role": "assistant", "content": assistant_payload})
            return result
        if not isinstance(messages, list):
            raise TypeError("messages must be a string, sequence, or a list of Message/Dict objects.")
        if messages and isinstance(messages[0], Message):
            return [{"role": m.role, "content": m.content} for m in messages]
        normalised: List[Dict[str, str]] = []
        for item in messages:
            if isinstance(item, Message):
                normalised.append({"role": item.role, "content": item.content})
            elif isinstance(item, dict):
                if "role" in item:
                    normalised.append({"role": item["role"], "content": item.get("content", "")})
                else:
                    normalised.extend(LLMChat._normalise_messages(item))
            elif isinstance(item, str):
                normalised.append({"role": "user", "content": item})
            else:
                normalised.extend(LLMChat._normalise_messages(item))
        return normalised

    def _legacy_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Fallback prompt renderer when chat_template is unavailable."""
        prompt_parts = [self.begin_of_text or getattr(self.tokenizer, "bos_token", "") or ""]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            start = self.start_header_id
            end = self.end_header_id
            eot = self.eot_id or getattr(self.tokenizer, "eos_token", "") or ""
            if start and end:
                prompt_parts.append(f"{start}{role}{end}\n{content}{eot}")
            else:
                prompt_parts.append(f"[{role.upper()}]\n{content}{eot}")
        if self.start_header_id and self.end_header_id:
            prompt_parts.append(f"{self.start_header_id}assistant{self.end_header_id}\n")
        else:
            prompt_parts.append("[ASSISTANT]\n")
        return "".join(prompt_parts)

    def _build_chat_inputs(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        assistant_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], str, int]:
        """Tokenize chat messages and return model inputs, text, and prompt length."""
        normalised = self._normalise_messages(messages)
        assistant_prompt = assistant_prompt or self.default_assistant_prompt
        prompt_text = ""
        try:
            template_kwargs = _chat_template_kwargs()
            apply_kwargs: Dict[str, Any] = {
                "add_generation_prompt": add_generation_prompt,
                "tokenize": False,
            }
            if template_kwargs:
                apply_kwargs["chat_template_kwargs"] = template_kwargs
            prompt_text = self.tokenizer.apply_chat_template(
                normalised,
                **apply_kwargs,
            ) + assistant_prompt

            tokenized = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)

            if isinstance(tokenized, dict):
                inputs = tokenized
            else:
                inputs = {
                    "input_ids": tokenized,
                    "attention_mask": torch.ones_like(tokenized),
                }
        except (ValueError, AttributeError, NotImplementedError, TypeError):

            prompt_text = self._legacy_prompt_from_messages(normalised) + assistant_prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        return inputs, prompt_text, input_length

    def _render_base_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """Render base messages from the current template with provided text."""
        rendered: List[Dict[str, str]] = []
        template = self.base_messages_template or [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        for block in template:
            content_template = block.get("content", "")
            content = content_template.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            rendered.append({"role": block.get("role", "user"), "content": content})
        return rendered

    def format_chat_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
    ) -> str:
        """Render a single chat block for the given role and content."""
        prefix = self.begin_of_text if include_begin else ""
        start = self.start_header_id
        end = self.end_header_id
        eot = self.eot_id if include_eot else ""
        if start and end:
            return f"{prefix}{start}{role}{end}\n{content}{eot}"
        upper_role = role.upper()
        return f"{prefix}[{upper_role}]\n{content}{eot}"

    def tokenize_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single chat segment and move tensors to the model device."""
        text = self.format_chat_segment(
            role,
            content,
            include_begin=include_begin,
            include_eot=include_eot,
        )
        tokens = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        return {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokens.items()
        }

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: Optional[str] = None,
        *,
        add_generation_prompt: bool = True,
        return_messages: bool = False,
    ) -> Dict[str, Any]:
        """Create model inputs from system/user prompts and optional assistant suffix."""
        messages = self._render_base_messages(system_prompt, user_prompt)
        inputs, prompt_text, prompt_length = self._build_chat_inputs(
            messages,
            assistant_prompt=assistant_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        result: Dict[str, Any] = {
            "inputs": inputs,
            "prompt_text": prompt_text,
            "prompt_length": prompt_length,
        }
        if return_messages:
            result["messages"] = messages
        return result

    @classmethod
    def _ensure_thread_pool(cls, workers: int) -> None:
        """Initialise or resize the shared thread pool used for CPU work."""
        if cls._THREAD_POOL is None or cls._THREAD_POOL_WORKERS != workers:
            if cls._THREAD_POOL is not None:
                cls._THREAD_POOL.shutdown(wait=False)
            cls._THREAD_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="LLM-chat")
            cls._THREAD_POOL_WORKERS = workers

    @classmethod
    def finalize_request(cls, request_uid: str) -> None:
        KVCOMMEngine.finalize_request(request_uid)

    def get_request_state(self, request_uid: str) -> "_RequestState":
        """Return per-request state used by the KV engine."""
        return self.kv_engine.get_request_state(request_uid)

    def _ensure_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Return the shared memory slot for a given agent id."""
        return LLMChat._ensure_shared_kv_memory().setdefault(agent_id, {})

    def _ensure_global_input_buckets(self) -> Dict[str, Dict[str, Any]]:
        """Ensure the global input buckets exist and return the shared store."""
        store = LLMChat._ensure_shared_kv_memory()
        store.setdefault("input", {})
        store.setdefault("input_ids", {})
        store.setdefault("input_drop_num", {})
        return store

    def has_prefix_initialized(self, agent_id: str) -> bool:
        """Check if prefix KV has been initialized for an agent."""
        if not LLMChat._initialization.get(agent_id, False):
            return False
        bucket = LLMChat._ensure_shared_kv_memory().get(agent_id) or {}
        if not isinstance(bucket, dict):
            return False
        prefix = bucket.get("prefix")
        if not prefix:
            return False
        # Static-only templates (e.g. clawbench bugfix without bench padding) have no
        # {agent_*}/{turn_*}/{user_question} slots — placeholder_info is {} but valid.
        return bucket.get("placeholder_info") is not None

    @staticmethod
    def _upstream_agent_index(ph_id: str) -> int | None:
        match = _AGENT_CURRENT_PLACEHOLDER_RE.match(str(ph_id))
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _upstream_response_kv_available(self, upstream_node: str, message_key: str) -> bool:
        bucket = LLMChat._shared_kv_cache_memory.get(str(upstream_node)) or {}
        if not isinstance(bucket, dict):
            return False
        resp = bucket.get("response") or {}
        if not isinstance(resp, dict):
            return False
        return bool(resp.get(message_key))

    def _kv_reuse_anchors_for_ph(
        self,
        ph_id: str,
        message: str,
        anchors_for_node: dict,
    ) -> list:
        """Return anchor list for kv_reuse blend; empty list = pass-through rotate only."""
        ph_id = str(ph_id)
        if _TURN_PLACEHOLDER_RE.match(ph_id):
            slot = self.resolve_turn_ph_slot(ph_id, message)
            if slot is not None and (slot.absolute_kv is not None or slot.kv_ref):
                return []
        upstream_idx = self._upstream_agent_index(ph_id)
        if upstream_idx is not None:
            try:
                node_idx = int(self.node_id)
            except (TypeError, ValueError):
                node_idx = -1
            slot = self.resolve_upstream_agent_slot(ph_id, message)
            if slot is not None and slot.materialization in ("consumer_contextual", "producer_contextual"):
                return []
            if upstream_idx < node_idx and self._upstream_response_kv_available(str(upstream_idx), message):
                return []
        bucket = anchors_for_node.get(ph_id)
        if not isinstance(bucket, dict):
            bucket = {}

        prefix_store = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        from sidecar.stores.prefix_spans import normalize_placeholder_info
        from sidecar.stores.topology_anchor import (
            delta_key_from_ph_rec,
            serialize_anchor_key,
        )

        ph_info = normalize_placeholder_info(prefix_store.get("placeholder_info"))
        ph_rec = ph_info.get(ph_id)
        if isinstance(ph_rec, dict):
            static_hash = str(prefix_store.get("static_template_hash") or "")
            topo = str(prefix_store.get("topology_id") or "")
            content_hash = self._upstream_agent_content_hash(ph_id, message) or ""
            delta_key = delta_key_from_ph_rec(
                ph_id=ph_id,
                ph_rec=ph_rec,
                static_template_hash=static_hash,
                topology_id=topo,
                content_hash=content_hash,
            )
            pool_entry = self._get_store_registry().agent_anchors.get_by_topology_key(
                node_id=str(self.node_id),
                message_key=str(message),
                delta_key=delta_key,
            )
            if pool_entry is not None and pool_entry.ph_delta is not None:
                topo_blob = serialize_anchor_key(delta_key)
                return [
                    {
                        f"{self.node_id}_ph_key_delta": pool_entry.ph_delta,
                        f"{self.node_id}_ph_value_delta": pool_entry.ph_value_delta,
                        f"{self.node_id}_pf_key_delta": pool_entry.pf_delta,
                        f"{self.node_id}_pf_value_delta": pool_entry.pf_value_delta,
                        "anchor_topology_key": topo_blob,
                    }
                ]

        if not bucket:
            return []
        return list(bucket.values())

    def placeholders_missing_anchor_delta(self, request_uid: str, message: str) -> List[str]:
        """Placeholder ids that still need a dense anchor materialization pass."""
        state = self.get_request_state(request_uid)
        ph_ids = (
            (LLMChat._ensure_shared_kv_memory().get(self.node_id) or {}).get("placeholder_info") or {}
        ).keys()
        missing: List[str] = []
        try:
            node_idx = int(self.node_id)
        except (TypeError, ValueError):
            node_idx = -1
        for ph_id in ph_ids:
            upstream_idx = self._upstream_agent_index(str(ph_id))
            slot = self.resolve_upstream_agent_slot(str(ph_id), message)
            if slot is not None and slot.materialization == "consumer_contextual":
                continue
            if (
                upstream_idx is not None
                and upstream_idx < node_idx
                and self._upstream_response_kv_available(str(upstream_idx), message)
            ):
                continue
            bucket = state.anchor_dict.setdefault(ph_id, {})
            if not isinstance(bucket, dict):
                continue
            anchor_entry = (state.anchors.get(ph_id) or {}) if isinstance(state.anchors, dict) else {}
            if not isinstance(anchor_entry, dict):
                anchor_entry = {}
            if bucket.get(message) and f"{self.node_id}_ph_key_delta" not in anchor_entry.get(message, {}):
                missing.append(str(ph_id))
        return missing

    def has_active_anchor(self, request_uid: str, message: str) -> bool:
        """Determine whether an anchor should trigger dense prefill."""
        return bool(self.placeholders_missing_anchor_delta(request_uid, message))

    def can_kv_reuse_with_turn_only_gaps(self, request_uid: str, message: str) -> bool:
        """Allow kv_reuse when only ephemeral OpenClaw turn placeholders lack anchor deltas."""
        missing = self.placeholders_missing_anchor_delta(request_uid, message)
        if not missing:
            return True
        return all(_TURN_PLACEHOLDER_RE.match(ph_id) for ph_id in missing)

    def can_kv_reuse_with_soft_anchor_gaps(self, request_uid: str, message: str) -> bool:
        """Allow kv_reuse when missing deltas are only ephemeral turn_* placeholders.

        Cross-agent ``agent_N_current`` slots must be dense-materialised on the
        consuming node; treating them as soft gaps causes silent blend skips when
        pooled pf/ph deltas no longer match the current prefix topology.
        """
        missing = self.placeholders_missing_anchor_delta(request_uid, message)
        if not missing:
            return True
        return all(_TURN_PLACEHOLDER_RE.match(ph_id) for ph_id in missing)

    @staticmethod
    def _delta_seq_len(delta: Any) -> int | None:
        if delta is None or not hasattr(delta, "shape") or len(delta.shape) < 2:
            return None
        return int(delta.shape[-2])

    def _merged_anchor_entry(self, request_uid: str, ph_id: str, message: str) -> Dict[str, Any] | None:
        """Return request-scoped anchor entry, falling back to committed global store."""
        state = self.get_request_state(request_uid)
        entry = (state.anchors.get(ph_id) or {}).get(message)
        if isinstance(entry, dict):
            return entry
        global_entry = (KVCOMMEngine.anchors.get(ph_id) or {}).get(message)
        if isinstance(global_entry, dict):
            return global_entry
        return None

    def _prefix_segment_len_for_placeholder(self, ph_id: str) -> int | None:
        """Length of the prefix KV segment paired with ``ph_id`` via pf_span_id."""
        from sidecar.stores.prefix_spans import (
            legacy_forward_pf_kv_index,
            normalize_placeholder_info,
            resolve_pf_kv_index,
        )

        prefix_store = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        if not isinstance(prefix_store, dict):
            return None
        placeholder_info = prefix_store.get("placeholder_info") or {}
        span_registry = prefix_store.get("span_registry") or {}
        prefix_kv_list = prefix_store.get("prefix") or []
        if not isinstance(placeholder_info, dict) or not placeholder_info or not prefix_kv_list:
            return None

        kv_idx = resolve_pf_kv_index(
            ph_id=str(ph_id),
            placeholder_info=placeholder_info,
            span_registry=span_registry,
        )
        if kv_idx is None:
            kv_idx = legacy_forward_pf_kv_index(placeholder_info, str(ph_id))
        if kv_idx is None or kv_idx >= len(prefix_kv_list):
            return None
        return int(prefix_kv_list[kv_idx]._seen_tokens)

    def _resolve_pf_for_ph(
        self,
        node_id: str,
        ph_id: str,
        *,
        prefix_store: dict | None = None,
        prefix_kv_list: list | None = None,
        prefix_token_ids: list | None = None,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Resolve pf KV/token ids for a placeholder via span registry or SegmentCache."""
        from sidecar.stores.prefix_spans import legacy_forward_pf_kv_index, resolve_pf_kv_index

        store = prefix_store if isinstance(prefix_store, dict) else (LLMChat._shared_kv_cache_memory.get(node_id) or {})
        kv_list = prefix_kv_list if prefix_kv_list is not None else (store.get("prefix") or [])
        tok_list = prefix_token_ids if prefix_token_ids is not None else (store.get("token_ids") or [])
        ph_info = store.get("placeholder_info") or {}
        span_registry = store.get("span_registry") or {}

        kv_idx = resolve_pf_kv_index(
            ph_id=str(ph_id),
            placeholder_info=ph_info,
            span_registry=span_registry,
        )
        if kv_idx is not None and 0 <= kv_idx < len(kv_list):
            tok = tok_list[kv_idx] if kv_idx < len(tok_list) else {}
            return kv_list[kv_idx], tok

        seg_cache = self._get_store_registry().segment_cache.for_node(str(node_id))
        resolved = seg_cache.resolve_pf(str(ph_id))
        if resolved is not None:
            return resolved

        legacy_idx = legacy_forward_pf_kv_index(ph_info, str(ph_id))
        if legacy_idx is not None and 0 <= legacy_idx < len(kv_list):
            tok = tok_list[legacy_idx] if legacy_idx < len(tok_list) else {}
            return kv_list[legacy_idx], tok

        raise RuntimeError(
            f"No pf segment resolved for placeholder '{ph_id}' on node '{node_id}' "
            f"(span_registry keys={list(span_registry.keys())})."
        )

    def find_incompatible_anchor_deltas(self, request_uid: str, message: str) -> List[str]:
        """Return placeholder ids whose stored deltas no longer match current prefix segments."""
        from sidecar.stores.topology_anchor import coordinate_shifted, current_topology_keys

        prefix_store = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        if not isinstance(prefix_store, dict):
            return []

        placeholder_info = prefix_store.get("placeholder_info") or {}
        if not isinstance(placeholder_info, dict) or not placeholder_info:
            return []

        current_keys = current_topology_keys(prefix_store)
        delta_key_ph = f"{self.node_id}_ph_key_delta"
        delta_key_pf = f"{self.node_id}_pf_key_delta"
        delta_key_pf_val = f"{self.node_id}_pf_value_delta"
        incompatible: List[str] = []

        try:
            node_idx = int(self.node_id)
        except (TypeError, ValueError):
            node_idx = -1

        for ph_id in placeholder_info:
            ph_id_str = str(ph_id)
            upstream_idx = self._upstream_agent_index(ph_id_str)
            if (
                upstream_idx is not None
                and upstream_idx < node_idx
                and self._upstream_response_kv_available(str(upstream_idx), message)
            ):
                continue

            entry = self._merged_anchor_entry(request_uid, ph_id_str, message)
            if not isinstance(entry, dict):
                continue
            if delta_key_ph not in entry and delta_key_pf not in entry:
                continue

            current_topo = current_keys.get(ph_id_str)
            stored_topo = entry.get("anchor_topology_key")
            if current_topo is not None and coordinate_shifted(stored_topo, current_topo):
                incompatible.append(ph_id_str)
                continue

            try:
                ph_cache, _, drop_num = self.kv_engine.fetch_shared_cache(ph_id_str, message)
            except RuntimeError:
                incompatible.append(ph_id_str)
                continue

            placeholder_len = int(ph_cache._seen_tokens - drop_num)
            ph_delta = entry.get(delta_key_ph)
            ph_delta_len = self._delta_seq_len(ph_delta)
            if ph_delta_len is not None and ph_delta_len != placeholder_len:
                incompatible.append(ph_id_str)
                continue

            if delta_key_pf in entry:
                pf_len = self._prefix_segment_len_for_placeholder(ph_id_str)
                pf_delta_len = self._delta_seq_len(entry.get(delta_key_pf))
                if pf_len is not None and pf_delta_len is not None and pf_delta_len != pf_len:
                    if ph_delta_len is not None and ph_delta_len == placeholder_len:
                        entry.pop(delta_key_pf, None)
                        entry.pop(delta_key_pf_val, None)
                        global_entry = (KVCOMMEngine.anchors.get(ph_id_str) or {}).get(message)
                        if isinstance(global_entry, dict):
                            global_entry.pop(delta_key_pf, None)
                            global_entry.pop(delta_key_pf_val, None)
                        continue
                    incompatible.append(ph_id_str)
        return incompatible

    def purge_incompatible_anchor_deltas(
        self,
        request_uid: str,
        message: str,
        ph_ids: List[str],
    ) -> None:
        """Drop stale per-node delta tensors so dense prefill can rematerialise anchors."""
        if not ph_ids:
            return
        state = self.get_request_state(request_uid)
        delta_keys = (
            f"{self.node_id}_ph_key_delta",
            f"{self.node_id}_ph_value_delta",
            f"{self.node_id}_pf_key_delta",
            f"{self.node_id}_pf_value_delta",
        )
        for ph_id in ph_ids:
            bucket = state.anchors.get(ph_id)
            if not isinstance(bucket, dict):
                continue
            entry = bucket.get(message)
            if not isinstance(entry, dict):
                continue
            for key in delta_keys:
                entry.pop(key, None)

    def resolve_generation_mode(self, request_uid: str, message: str, preferred_mode: Optional[str]) -> str:
        """Pick dense_prefill vs kv_reuse with stale-delta protection for bench sidecar."""
        if preferred_mode == "dense_prefill":
            return "dense_prefill"
        if preferred_mode != "kv_reuse":
            return "dense_prefill" if self.has_active_anchor(request_uid, message) else "kv_reuse"

        incompatible = self.find_incompatible_anchor_deltas(request_uid, message)
        if incompatible:
            self.purge_incompatible_anchor_deltas(request_uid, message, incompatible)
            logger.debug(
                "generate_for_agent stale anchor deltas node_id={} placeholders={} -> dense_prefill",
                self.node_id,
                incompatible,
            )
            return "dense_prefill"

        if not self.has_active_anchor(request_uid, message) or self.can_kv_reuse_with_soft_anchor_gaps(
            request_uid,
            message,
        ):
            return "kv_reuse"
        return "dense_prefill"

    def update_condition_anchor(
        self,
        *,
        request_uid: str,
        owner_agent_id: str,
        message: str,
        content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: Optional[str] = None,
        max_length: int = None,
    ) -> bool:
        """Materialise condition KV cache for another agent and update anchors."""
        state = self.get_request_state(request_uid)
        anchor_key = anchor_namespace or f"condition_{owner_agent_id}_current"

        owner_memory = self._ensure_agent_memory(owner_agent_id)
        condition_bucket = owner_memory.setdefault("condition", {})
        if message in condition_bucket:

            return state.anchor_dict.setdefault(anchor_key, {}).get(message, False)

        token_ids = self.tokenize_segment(
            role=role,
            content=content,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )
        if "position_ids" not in token_ids:
            position_ids = torch.arange(token_ids["input_ids"].shape[-1]).unsqueeze(0)
            token_ids["position_ids"] = position_ids.to(self.model.device)
        else:
            token_ids["position_ids"] = token_ids["position_ids"].to(self.model.device)
        token_ids["input_ids"] = token_ids["input_ids"].to(self.model.device)
        token_ids["attention_mask"] = token_ids["attention_mask"].to(self.model.device)

        prefix_ids = self.tokenize_segment(
            role=role,
            content=prefix_text,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )["input_ids"]
        drop_num = prefix_ids.shape[-1]

        if max_length is not None:
            token_ids["input_ids"] = token_ids["input_ids"][:, :drop_num + max_length]
            token_ids["attention_mask"] = token_ids["attention_mask"][:, :drop_num + max_length]
            token_ids["position_ids"] = token_ids["position_ids"][:, :drop_num + max_length]
            
        generated = self.model.generate(
            **token_ids,
            use_cache=True,
            max_length=token_ids["input_ids"].shape[-1] + 1,
            return_dict_in_generate=True,
            return_legacy_cache=False,
        )
        condition_cache = generated.past_key_values


        for key_name, value in (
            ("condition", condition_cache),
            ("condition_ids", token_ids),
            ("condition_drop_num", drop_num),
        ):
            bucket = owner_memory.setdefault(key_name, {})
            bucket.setdefault(message, []).append(value)

        anchor_store = state.anchors.setdefault(anchor_key, {})
        cond_anchor_list = list(anchor_store.values())
        cond_len_bucket = state.anchor_len_dict.setdefault(anchor_key, {})
        anchor_len_list = [
            cond_len_bucket.get(entry_key, [0, 0])
            for entry_key in anchor_store.keys()
        ]
        cond_info_bucket = state.anchor_info_dict.setdefault(anchor_key, {})
        anchor_activated_list = list(cond_info_bucket.values())

        total_prefix_len = 0
        for bucket in state.anchor_len_dict.values():
            if not isinstance(bucket, dict):
                continue
            length_entry = bucket.get(message, [0, 0])
            if isinstance(length_entry, (list, tuple)) and length_entry:
                total_prefix_len += length_entry[0]

        prob, anchor_activated_list = self.kv_engine.predict_as_anchor(
            condition_cache.copy().slice_(start=drop_num),
            anchor_kv_cache_list=cond_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_activated_list,
        )

        cond_flag_bucket = state.anchor_dict.setdefault(anchor_key, {})
        cond_flag_bucket[message] = prob

        global_bucket = state.global_anchor_info.setdefault(anchor_key, {})
        if not prob:
            info_items = list(cond_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                cond_info_bucket[msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
        else:
            cond_info_bucket[message] = 0
            global_bucket[message] = [
                0,
                condition_cache.get_seq_length() - drop_num,
            ]
        return prob

    def update_input_anchor(
        self,
        *,
        request_uid: str,
        agent_id: str,
        message: str,
        user_content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: str = "user_question",
        test_time: bool = False,
    ) -> str:
        """Ensure the user input placeholder cache is ready and choose a strategy."""
        state = self.get_request_state(request_uid)
        shared_mem = LLMChat._shared_kv_cache_memory
        agent_memory = self._ensure_agent_memory(agent_id)
        placeholder_info = agent_memory.get("placeholder_info")
        safe_message = _escape_loguru_markup(message)

        def _record_input_anchor_metrics(
            mode: str,
            *,
            pooled_tokens: int = 0,
            prediction: bool = False,
        ) -> str:
            state.input_anchor_metrics = {
                "input_routing_mode": mode,
                "input_anchor_prediction": prediction,
                "input_anchor_pooled_tokens": int(pooled_tokens),
            }
            return mode

        if message in (shared_mem.get("input") or {}):
            if not placeholder_info:
                logger.opt(colors=True).debug(
                    f"<yellow>No placeholder info found for agent '{agent_id}' while reusing input cache; "
                    f"prefix will be rebuilt on next prepare.</yellow>"
                )
                _log_input_anchor_routing(
                    node_id=str(self.node_id),
                    message_key=message,
                    placeholder_id="*",
                    reuse_kind="input_cache_no_ph_info",
                    routing_mode="kv_reuse",
                )
                return _record_input_anchor_metrics("kv_reuse")
            placeholder_entries = list(placeholder_info.items())[::-1]
            for ph_id, _ in placeholder_entries:
                bucket = state.anchor_dict.setdefault(ph_id, {})
                if bucket.get(message):
                    anchor_entry = state.anchors.get(ph_id) or {}
                    if not isinstance(anchor_entry, dict):
                        anchor_entry = {}
                    if f"{self.node_id}_ph_key_delta" in anchor_entry.get(message, {}):
                        _log_input_anchor_routing(
                            node_id=str(self.node_id),
                            message_key=message,
                            placeholder_id=str(ph_id),
                            reuse_kind="anchor_delta",
                            routing_mode="kv_reuse",
                        )
                        return _record_input_anchor_metrics("kv_reuse")
                    _log_input_anchor_routing(
                        node_id=str(self.node_id),
                        message_key=message,
                        placeholder_id=str(ph_id),
                        reuse_kind="anchor_flag_without_delta",
                        routing_mode="dense_prefill",
                    )
                    return _record_input_anchor_metrics("dense_prefill")
            _log_input_anchor_routing(
                node_id=str(self.node_id),
                message_key=message,
                placeholder_id="*",
                reuse_kind="input_cache",
                routing_mode="kv_reuse",
            )
            return _record_input_anchor_metrics("kv_reuse")

        token_ids = self.tokenize_segment(
            role=role,
            content=user_content,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )
        if "position_ids" in token_ids:
            position_ids = token_ids["position_ids"]
        else:
            position_ids = torch.arange(token_ids["input_ids"].shape[-1], dtype=torch.long)
        token_ids["position_ids"] = position_ids.unsqueeze(0).to(self.model.device)
        token_ids["input_ids"] = token_ids["input_ids"].to(self.model.device)
        token_ids["attention_mask"] = token_ids["attention_mask"].to(self.model.device)

        prefix_ids = self.tokenize_segment(
            role=role,
            content=prefix_text,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )["input_ids"]
        drop_num = prefix_ids.shape[-1]
        if test_time:
            for _ in range(10):
                if _ == 5:
                    torch.cuda.synchronize()
                    start_time = perf_counter()
                output = self.model.generate(
                    **token_ids,
                    use_cache=True,
                    max_length=token_ids["input_ids"].shape[-1] + 1,
                    return_dict_in_generate=True,
                    return_legacy_cache=False,
                )
            torch.cuda.synchronize()
            end_time = perf_counter()
            logger.opt(colors=True).info(
                f"<cyan>Latency for computing the input kv-cache of {message}: {(end_time - start_time) / 5:.3f} seconds</cyan>"
            )
        else:
            output = self.model.generate(
                **token_ids,
                use_cache=True,
                max_length=token_ids["input_ids"].shape[-1] + 1,
                return_dict_in_generate=True,
                return_legacy_cache=False,
            )
        input_cache = output.past_key_values

        global_buckets = self._ensure_global_input_buckets()
        global_buckets["input"].setdefault(message, []).append(
            input_cache.copy().slice_(start=0, end=token_ids["input_ids"].shape[-1])
        )
        global_buckets["input_ids"].setdefault(message, []).append(token_ids)
        global_buckets["input_drop_num"].setdefault(message, []).append(drop_num)

        anchor_store = state.anchors.setdefault(anchor_namespace, {})
        input_anchor_list = list(anchor_store.values())
        uq_len_bucket = state.anchor_len_dict.setdefault(anchor_namespace, {})
        anchor_len_list = [
            uq_len_bucket.get(entry_key, [0, 0])
            for entry_key in anchor_store.keys()
        ]
        uq_info_bucket = state.anchor_info_dict.setdefault(anchor_namespace, {})
        anchor_activated_list = list(uq_info_bucket.values())

        accumulate_len = 0
        for bucket in state.anchor_len_dict.values():
            if not isinstance(bucket, dict):
                continue
            length_entry = bucket.get(message, [0, 0])
            if isinstance(length_entry, (list, tuple)) and length_entry:
                accumulate_len += length_entry[0]

        prob, anchor_activated_list = self.kv_engine.predict_as_anchor(
            input_cache.copy().slice_(start=drop_num),
            anchor_kv_cache_list=input_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_activated_list,
            test_time=test_time,
        )
        logger.opt(colors=True).debug(
            f"<magenta>Anchor prediction for input '{safe_message}'</magenta>: {prob}"
        )

        state.anchor_dict.setdefault(anchor_namespace, {})[message] = prob
        global_bucket = state.global_anchor_info.setdefault(anchor_namespace, {})
        if not prob:
            info_items = list(uq_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                uq_info_bucket[msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
            return _record_input_anchor_metrics("kv_reuse")

        pooled_tokens = int(input_cache.get_seq_length() - drop_num)
        uq_info_bucket[message] = 0
        global_bucket[message] = [
            0,
            pooled_tokens,
        ]
        return _record_input_anchor_metrics(
            "dense_prefill",
            pooled_tokens=pooled_tokens,
            prediction=True,
        )

    async def generate_for_agent(
        self,
        *,
        request_uid: str,
        message: str,
        preferred_mode: Optional[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        on_token: Optional[Any] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response using the requested strategy with sensible fallbacks."""
        latency_target = output_dir or kwargs.get("output_dir")
        missing_delta = self.placeholders_missing_anchor_delta(request_uid, message)
        mode = self.resolve_generation_mode(request_uid, message, preferred_mode)
        logger.debug(
            "generate_for_agent routing: preferred_mode={} active_anchor={} missing_delta={} "
            "chosen_mode={} request_uid={} node_id={} message_key={}",
            preferred_mode,
            self.has_active_anchor(request_uid, message),
            missing_delta,
            mode,
            request_uid,
            self.node_id,
            _short_message_key(message),
        )

        stream_kwargs = {**kwargs, "on_token": on_token} if on_token is not None else kwargs
        if mode == "dense_prefill":
            return await self.generate_with_dense_prefill(
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                max_anchor_num=kwargs.get("max_anchor_num", self.config.max_anchor_num),
                window_length=kwargs.get("window_length", self.config.window_size),
                request_uid=request_uid,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=latency_target,
                **stream_kwargs,
            )
        return await self.generate_with_kv_reuse(
            message,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=latency_target,
            **stream_kwargs,
        )

    def _map_in_pool(self, fn, iterable, timeout=None):
        pool = LLMChat._THREAD_POOL
        if pool is None:
            raise RuntimeError("Thread pool not initialized")
        task_timeout = timeout or self.config.worker_timeout
        futures = [pool.submit(fn, *args) for args in iterable]
        for fut in as_completed(futures, timeout=task_timeout):
            try:
                yield fut.result(timeout=self.config.worker_timeout)
            except TimeoutError as exc:
                raise TimeoutError("Thread task timeout") from exc
            except Exception as exc:
                raise RuntimeError("Thread task failed") from exc

    def set_id(self, node_id: str, role: str):
        """Bind the chat instance to a graph node id and role."""
        self.node_id = node_id
        self.role = role

        bucket = LLMChat._shared_kv_cache_memory.get(self.node_id)
        if not isinstance(bucket, dict):
            LLMChat._shared_kv_cache_memory[self.node_id] = {}
            LLMChat._initialization[self.node_id] = False

    async def prepare_prefix_kv_segments(self, node_id: str, prefix: str, user_prompt: str):
        """Materialize and store prefix KV segments and placeholder indices.

        The rendered prompt is tokenized and executed once to obtain the KV
        cache of each text segment. These are stored in shared memory keyed by
        `node_id` for reuse during subsequent generations.
        """
        from sidecar.stores.prefix_spans import build_layout_from_segments

        messages = self._render_base_messages(prefix, user_prompt)
        _, prompt_text, _ = self._build_chat_inputs(messages, add_generation_prompt=True)
        _, token_ids, segments = self.locate_placeholder(prompt_text, return_segments=True)
        layout = build_layout_from_segments(segments)

        LLMChat._shared_kv_cache_memory.setdefault(node_id, {})

        def _prefill_prefix_kv():
            with torch.no_grad():
                with LLMChat._model_lock:
                    return self.model.generate(
                        **token_ids,
                        use_cache=True,
                        max_length=token_ids["input_ids"].shape[-1] + 1,
                        return_dict_in_generate=True,
                        return_legacy_cache=False,
                    )

        logger.info(
            "Building prefix KV for node {} ({} prompt chars; first HF prefill may take 1-3 min on 32B)",
            node_id,
            len(prompt_text),
        )
        out = await asyncio.to_thread(_prefill_prefix_kv)
        logger.info("Prefix KV ready for node {}", node_id)
        prompt_len = int(token_ids["input_ids"].shape[-1])
        base_kv = out.past_key_values.slice_(start=0, end=prompt_len)

        segment_kv_list, token_id_list, ph_info, span_registry, span_order = self._materialize_prefix_segments(
            layout,
            base_kv,
            segments,
        )
        turn_count = self._infer_turn_count_from_placeholders(ph_info)
        self._commit_prefix_bucket(
            node_id,
            segment_kv_list=segment_kv_list,
            token_id_list=token_id_list,
            placeholder_info=ph_info,
            span_registry=span_registry,
            prefix_span_order=span_order,
            prompt_token_len=prompt_len,
            prompt_input_ids=token_ids["input_ids"],
            base_kv_full=base_kv,
            system_prompt=prefix,
            user_template=user_prompt,
            turn_count=turn_count,
            segments=segments,
            prompt_text=prompt_text,
        )

        self._initialization[node_id] = LLMChat._initialization[node_id] = True

    def get_prefix_turn_count(self, node_id: str) -> int:
        bucket = LLMChat._shared_kv_cache_memory.get(node_id, {})
        try:
            return int(bucket.get("turn_count", 0))
        except (TypeError, ValueError):
            return 0

    def set_prefix_turn_count(self, node_id: str, turn_count: int) -> None:
        bucket = LLMChat._shared_kv_cache_memory.setdefault(node_id, {})
        bucket["turn_count"] = int(turn_count)

    @staticmethod
    def _infer_turn_count_from_placeholders(placeholder_info: dict) -> int:
        turn_indices: set[int] = set()
        for ph_id in placeholder_info or {}:
            if not str(ph_id).startswith("turn_"):
                continue
            parts = str(ph_id).split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                turn_indices.add(int(parts[1]))
        return max(turn_indices) if turn_indices else 0

    @staticmethod
    def _get_store_registry():
        from sidecar.stores.registry import get_store_registry

        return get_store_registry()

    def _sync_prefix_stores(
        self,
        node_id: str,
        *,
        prefix_kv_list: list,
        prefix_token_ids: list,
        placeholder_info: dict,
        user_template: str,
        turn_count: int,
        span_registry: dict | None = None,
        prefix_span_order: list[str] | None = None,
    ) -> None:
        from sidecar.stores.hashing import static_template_hash, topology_id
        from sidecar.stores.prefix_topology import write_topology

        bucket = LLMChat._shared_kv_cache_memory.setdefault(node_id, {})
        static_hash = static_template_hash(user_template)
        topo = topology_id(static_hash=static_hash, turn_count=int(turn_count))
        write_topology(bucket, user_template=user_template, turn_count=int(turn_count))

        stores = self._get_store_registry()
        stores.segment_cache.for_node(node_id).put_prefix_blob(
            prefix_kv_list=prefix_kv_list,
            prefix_token_ids=prefix_token_ids,
            placeholder_info=placeholder_info,
            static_template_hash=static_hash,
            topology_id=topo,
            turn_count=int(turn_count),
            span_registry=span_registry or bucket.get("span_registry") or {},
            prefix_span_order=prefix_span_order or bucket.get("prefix_span_order") or [],
        )

    @staticmethod
    def _text_encodings_in_order(segments: list[tuple]) -> list[dict[str, torch.Tensor]]:
        return [seg[2] for seg in segments if seg[0] == "text"]

    def _materialize_prefix_segments(
        self,
        layout,
        base_kv,
        segments: list[tuple],
        *,
        old_bucket: dict | None = None,
        frozen_count: int = 0,
        extend_placeholders_only: bool = False,
    ) -> tuple[list, list, dict, dict, list]:
        """Build prefix KV lists from layout; reuse frozen left segments when requested."""
        from sidecar.stores.prefix_spans import normalize_placeholder_info

        old_bucket = old_bucket or {}
        old_prefix = list(old_bucket.get("prefix") or [])
        old_token_ids = list(old_bucket.get("token_ids") or [])
        old_registry = dict(old_bucket.get("span_registry") or {})
        text_encodings = self._text_encodings_in_order(segments)

        segment_kv_list: list = []
        token_id_list: list = []
        span_registry: dict = {}

        for i, span in enumerate(layout.text_spans):
            span_id = span.span_id
            old_entry = old_registry.get(span_id)
            reuse = (
                bool(old_prefix)
                and old_entry is not None
                and i < frozen_count
                and old_entry.get("text_hash") == span.text_hash
                and int(old_entry.get("token_start", -1)) == span.token_start
                and int(old_entry.get("token_end", -1)) == span.token_end
            )
            if reuse:
                kv_idx = int(old_entry.get("kv_index", i))
                segment_kv_list.append(old_prefix[kv_idx])
                token_id_list.append(old_token_ids[kv_idx] if kv_idx < len(old_token_ids) else text_encodings[i])
            else:
                segment_kv_list.append(base_kv.slice(start=span.token_start, end=span.token_end))
                token_id_list.append(text_encodings[i] if i < len(text_encodings) else {})
            kv_index = len(segment_kv_list) - 1
            span_registry[span_id] = {
                "span_id": span_id,
                "text_hash": span.text_hash,
                "token_start": span.token_start,
                "token_end": span.token_end,
                "kv_index": kv_index,
            }

        if extend_placeholders_only:
            ph_info = dict(normalize_placeholder_info(old_bucket.get("placeholder_info")))
            for ph_id, rec in layout.placeholder_info.items():
                if ph_id not in ph_info:
                    ph_info[ph_id] = dict(rec)
        else:
            ph_info = dict(layout.placeholder_info)

        return (
            segment_kv_list,
            token_id_list,
            ph_info,
            span_registry,
            list(layout.prefix_span_order),
        )

    def _populate_template_ph_bases(
        self,
        node_id: str,
        *,
        base_kv_full,
        placeholder_info: dict,
        segments: list | None = None,
    ) -> None:
        """Slice pure template ph KV from base_kv_full and store by topology coordinates."""
        from sidecar.stores.hashing import static_template_hash, topology_id
        from sidecar.stores.prefix_spans import normalize_placeholder_info
        from sidecar.stores.template_ph_base import TemplatePhBaseRecord

        bucket = LLMChat._shared_kv_cache_memory.get(node_id) or {}
        static_hash = str(bucket.get("static_template_hash") or "")
        topo = str(bucket.get("topology_id") or "")
        if not static_hash:
            user_template = str(bucket.get("user_template") or "")
            turn_count = int(bucket.get("turn_count") or 0)
            static_hash = static_template_hash(user_template)
            topo = topology_id(static_hash=static_hash, turn_count=turn_count)

        ph_info = normalize_placeholder_info(placeholder_info)
        seg_by_ph: dict[str, dict] = {}
        if segments:
            for seg in segments:
                if seg[0] == "placeholder":
                    seg_by_ph[str(seg[1])] = seg[2]

        records: dict[str, TemplatePhBaseRecord] = {}
        for ph_id, rec in ph_info.items():
            start = int(rec.get("start", 0))
            end = int(rec.get("end", 0))
            if start >= end or base_kv_full is None:
                continue
            records[str(ph_id)] = TemplatePhBaseRecord(
                ph_id=str(ph_id),
                static_template_hash=static_hash,
                topology_id=topo,
                ph_token_start=start,
                ph_token_end=end,
                pf_span_id=rec.get("pf_span_id"),
                absolute_kv=base_kv_full.slice(start=start, end=end),
                token_ids=seg_by_ph.get(str(ph_id)) or {},
            )
        self._get_store_registry().template_ph_base.replace_node(str(node_id), records)

    def _commit_prefix_bucket(
        self,
        node_id: str,
        *,
        segment_kv_list: list,
        token_id_list: list,
        placeholder_info: dict,
        span_registry: dict,
        prefix_span_order: list[str],
        prompt_token_len: int,
        prompt_input_ids: torch.Tensor,
        base_kv_full,
        system_prompt: str,
        user_template: str,
        turn_count: int,
        segments: list | None = None,
        prompt_text: str | None = None,
    ) -> None:
        bucket = LLMChat._shared_kv_cache_memory.setdefault(node_id, {})
        prev_ph_info = dict(bucket.get("placeholder_info") or {})
        bucket["prefix"] = segment_kv_list
        bucket["token_ids"] = token_id_list
        bucket["placeholder_info"] = placeholder_info
        bucket["span_registry"] = span_registry
        bucket["prefix_span_order"] = prefix_span_order
        bucket["prompt_token_len"] = int(prompt_token_len)
        bucket["prompt_input_ids"] = prompt_input_ids.detach().clone()
        bucket["base_kv_full"] = base_kv_full
        bucket["system_prompt"] = system_prompt
        bucket["user_template"] = user_template
        bucket.pop("turn", None)
        bucket["turn_count"] = int(turn_count)
        bucket["_prev_placeholder_info"] = prev_ph_info
        if prompt_text is not None:
            bucket["prompt_text"] = str(prompt_text)
        self._sync_prefix_stores(
            node_id,
            prefix_kv_list=segment_kv_list,
            prefix_token_ids=token_id_list,
            placeholder_info=placeholder_info,
            user_template=user_template,
            turn_count=turn_count,
            span_registry=span_registry,
            prefix_span_order=prefix_span_order,
        )
        self._populate_template_ph_bases(
            node_id,
            base_kv_full=base_kv_full,
            placeholder_info=placeholder_info,
            segments=segments,
        )

    def _prefill_suffix_on_past_sync(
        self,
        past_kv,
        token_ids: dict[str, torch.Tensor],
        past_len: int,
    ):
        """Forward only the suffix tokens on an existing prefix KV cache."""
        suffix_ids = token_ids["input_ids"][:, past_len:]
        if suffix_ids.shape[-1] <= 0:
            return past_kv
        attn = token_ids.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(token_ids["input_ids"])
        with torch.no_grad():
            with LLMChat._model_lock:
                output = self.model(
                    input_ids=suffix_ids,
                    attention_mask=attn,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
        return output.past_key_values

    async def append_prefix_segment_incremental(
        self,
        node_id: str,
        segment_template: str,
        *,
        system_prompt: str | None = None,
        expected_user_template: str | None = None,
    ) -> None:
        """Append a turn segment reusing frozen left prefix KV; forward suffix only."""
        from sidecar.stores.prefix_spans import (
            build_layout_from_segments,
            frozen_span_count,
            shared_prefix_token_len,
        )

        from sidecar.openclaw_prefix import merge_turn_segment_into_user_template

        bucket = LLMChat._shared_kv_cache_memory.get(node_id) or {}
        stored_user = str(bucket.get("user_template") or "").strip()
        stored_system = str(
            bucket.get("system_prompt") or system_prompt or ""
        ).strip()
        segment = str(segment_template or "")

        # Always extend the committed template in-bucket; openclaw re-parse may
        # reshape whitespace/static text and break prefix token identity.
        merged_user = merge_turn_segment_into_user_template(
            stored_user,
            segment,
            expected_user_template=expected_user_template,
        )

        messages = self._render_base_messages(stored_system, merged_user)
        _, prompt_text, _ = self._build_chat_inputs(messages, add_generation_prompt=True)
        _, token_ids, segments = self.locate_placeholder(prompt_text, return_segments=True)
        layout = build_layout_from_segments(segments)

        old_registry = dict(bucket.get("span_registry") or {})
        old_span_order = list(bucket.get("prefix_span_order") or [])
        old_token_len = int(bucket.get("prompt_token_len") or 0)
        old_input_ids = bucket.get("prompt_input_ids")
        base_kv_full = bucket.get("base_kv_full")
        old_prefix = bucket.get("prefix") or []

        frozen = frozen_span_count(old_registry, old_span_order, layout)
        new_prompt_len = int(token_ids["input_ids"].shape[-1])
        prefix_reuse_len = 0
        if isinstance(old_input_ids, torch.Tensor) and old_token_len > 0:
            device = token_ids["input_ids"].device
            prefix_reuse_len = shared_prefix_token_len(
                old_input_ids.to(device),
                token_ids["input_ids"],
            )

        min_reuse_len = max(32, int(old_token_len * 0.5))
        blockers: list[str] = []
        if not old_prefix:
            blockers.append("missing_old_prefix")
        if base_kv_full is None:
            blockers.append("missing_base_kv_full")
        if old_token_len <= 0:
            blockers.append("missing_old_token_len")
        if new_prompt_len <= prefix_reuse_len:
            blockers.append("suffix_not_longer")
        if prefix_reuse_len < min_reuse_len:
            blockers.append(
                f"shared_prefix_too_short={prefix_reuse_len}_min={min_reuse_len}"
            )
        if len(old_span_order) <= 0:
            blockers.append("empty_old_span_order")

        can_incremental = not blockers

        if not can_incremental:
            logger.debug(
                "[kvcomm-prefix] incremental append fallback node={} blockers={} "
                "old_tokens={} new_tokens={} shared_prefix={} stored_user_chars={} merged_user_chars={}",
                node_id,
                blockers,
                old_token_len,
                new_prompt_len,
                prefix_reuse_len,
                len(stored_user),
                len(merged_user),
            )
            await self.prepare_prefix_kv_segments(node_id, stored_system, merged_user)
            return

        suffix_tokens = new_prompt_len - prefix_reuse_len
        logger.info(
            "Incremental prefix append node {} (shared_prefix={}/{} frozen_spans={}/{} suffix_tokens={})",
            node_id,
            prefix_reuse_len,
            old_token_len,
            frozen,
            len(old_span_order),
            suffix_tokens,
        )

        past_kv = base_kv_full.slice(start=0, end=prefix_reuse_len)

        def _forward_suffix():
            return self._prefill_suffix_on_past_sync(past_kv, token_ids, prefix_reuse_len)

        new_base_kv = await asyncio.to_thread(_forward_suffix)
        new_base_kv = new_base_kv.slice_(start=0, end=new_prompt_len)

        segment_kv_list, token_id_list, ph_info, span_registry, span_order = self._materialize_prefix_segments(
            layout,
            new_base_kv,
            segments,
            old_bucket=bucket,
            frozen_count=frozen,
            extend_placeholders_only=True,
        )
        turn_count = self._infer_turn_count_from_placeholders(ph_info)
        self._commit_prefix_bucket(
            node_id,
            segment_kv_list=segment_kv_list,
            token_id_list=token_id_list,
            placeholder_info=ph_info,
            span_registry=span_registry,
            prefix_span_order=span_order,
            prompt_token_len=new_prompt_len,
            prompt_input_ids=token_ids["input_ids"],
            base_kv_full=new_base_kv,
            system_prompt=stored_system,
            user_template=merged_user,
            turn_count=turn_count,
            segments=segments,
            prompt_text=prompt_text,
        )
        self._initialization[node_id] = LLMChat._initialization[node_id] = True

    def resolve_turn_ph_slot(self, ph_id: str, message_key: str):
        return self._get_store_registry().turn_slots.get(
            str(self.node_id),
            str(message_key),
            str(ph_id),
        )

    def resolve_upstream_agent_slot(self, ph_id: str, message_key: str):
        """Resolve consumer- or producer-contextual slot for ``agent_*_current``."""
        from sidecar.stores.hashing import sha256_text

        stores = self._get_store_registry()
        ph_id = str(ph_id)
        message_key = str(message_key)

        content_hash = self._upstream_agent_content_hash(ph_id, message_key)
        if content_hash:
            consumer = stores.upstream_agent_slots.get_consumer(
                str(self.node_id),
                message_key,
                ph_id,
                content_hash,
            )
            if consumer is not None:
                return consumer

        upstream_idx = self._upstream_agent_index(ph_id)
        if upstream_idx is not None and content_hash:
            producer = stores.upstream_agent_slots.get_producer(
                str(upstream_idx),
                message_key,
                ph_id,
                content_hash,
            )
            if producer is not None:
                return producer
        return None

    def _upstream_agent_content_hash(self, ph_id: str, message_key: str) -> str | None:
        """Content hash for upstream agent output referenced by ``ph_id``."""
        from sidecar.stores.hashing import sha256_text

        upstream_idx = self._upstream_agent_index(str(ph_id))
        if upstream_idx is None:
            return None
        bucket = LLMChat._shared_kv_cache_memory.get(str(upstream_idx)) or {}
        if not isinstance(bucket, dict):
            return None
        ids_bucket = bucket.get("response_ids") or {}
        if not isinstance(ids_bucket, dict):
            return None
        values = ids_bucket.get(message_key)
        if not values:
            return None
        try:
            entry = values[-1]
        except (TypeError, IndexError):
            return None
        if not isinstance(entry, dict) or entry.get("input_ids") is None:
            return None
        text = self.tokenizer.decode(entry["input_ids"][0], skip_special_tokens=True)
        text = str(text).strip() or " "
        return sha256_text(text)

    def _export_producer_contextual_response_kv(
        self,
        *,
        full_kv_cache,
        prefix_token_len: int,
        response_token_ids: torch.Tensor,
        message: str,
        stored_text: str,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Slice response KV from full generation cache (producer-contextual)."""
        from sidecar.stores.hashing import sha256_text

        resp_len = int(response_token_ids.shape[-1])
        if resp_len <= 0:
            return self._forward_text_to_kv_sync(stored_text or " ")

        end = int(prefix_token_len) + resp_len
        response_kv_cache = full_kv_cache.slice_(start=int(prefix_token_len), end=end)
        response_kv_cache = self.kv_engine.apply_rotary_pos_emb(
            response_kv_cache,
            offset=-int(prefix_token_len),
            drop_num=0,
        )
        response_tokens = response_token_ids.unsqueeze(0) if response_token_ids.dim() == 1 else response_token_ids
        token_dict: Dict[str, torch.Tensor] = {
            "input_ids": response_tokens.to(self.model.device),
            "attention_mask": torch.ones_like(response_tokens).to(self.model.device),
        }
        content_hash = sha256_text(str(stored_text).strip() or " ")
        ph_id = f"agent_{self.node_id}_current"
        self._get_store_registry().upstream_agent_slots.put_producer(
            producer_node_id=str(self.node_id),
            message_key=str(message),
            ph_id=ph_id,
            content_hash=content_hash,
            absolute_kv=response_kv_cache,
            token_ids=token_dict,
            prefix_token_len=int(prefix_token_len),
        )
        return response_kv_cache, token_dict

    def _materialize_consumer_upstream_slot_sync(
        self,
        *,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
        slot_token_start: int,
        ph_token_ids: Dict[str, torch.Tensor],
        content_hash: str,
        upstream_node_id: str,
        drop_num: int = 0,
    ) -> bool:
        """Forward upstream payload on consumer frozen prefix left of the placeholder."""
        bucket = LLMChat._shared_kv_cache_memory.get(str(consumer_node_id)) or {}
        base_kv_full = bucket.get("base_kv_full")
        if base_kv_full is None:
            return False

        input_ids = ph_token_ids.get("input_ids")
        if input_ids is None or int(input_ids.shape[-1]) <= 0:
            return False

        slot_start = int(slot_token_start)
        real_ids = input_ids[:, drop_num:] if drop_num else input_ids
        real_len = int(real_ids.shape[-1])
        if real_len <= 0:
            return False

        attn_len = slot_start + real_len
        attention_mask = torch.ones(1, attn_len, dtype=torch.long, device=self.model.device)

        with torch.no_grad():
            with LLMChat._model_lock:
                past_kv = base_kv_full.slice(start=0, end=slot_start)
                output = self.model(
                    input_ids=real_ids.to(self.model.device),
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
        full_kv = output.past_key_values
        ctx_slice = full_kv.slice(start=slot_start, end=slot_start + real_len)
        ctx_relative = self.kv_engine.apply_rotary_pos_emb(
            ctx_slice,
            offset=-slot_start,
            drop_num=0,
        )
        token_dict = {
            "input_ids": real_ids.to(self.model.device),
            "attention_mask": torch.ones_like(real_ids).to(self.model.device),
        }
        self._get_store_registry().upstream_agent_slots.put_consumer(
            consumer_node_id=str(consumer_node_id),
            message_key=str(message_key),
            ph_id=str(ph_id),
            content_hash=str(content_hash),
            absolute_kv=ctx_relative,
            token_ids=token_dict,
            upstream_node_id=str(upstream_node_id),
            slot_token_start=slot_start,
            drop_num=0,
        )
        return True

    def _resolve_base_caches_for_dense_anchor(
        self,
        ph_id_list: List[str],
        merged_prefix_kv,
        placeholder_indices: Dict[str, Tuple[int, int]],
    ) -> Tuple[List[Any], List[Any]]:
        """Build base ph/pf lists for dense set_anchor.

        ph base must match real ph length (materialized merged prefix), not the short
        template slot slice. Template ph KV lives in ``template_ph_base`` for topology
        coordinates only. pf base comes from frozen text segments via ``pf_span_id``.
        """
        prefix_store = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        prefix_kv_list = prefix_store.get("prefix") or []

        base_ph_list, _split_pf_list = merged_prefix_kv.split_cache_by_placeholders(
            placeholder_indices
        )
        base_pf_list: List[Any] = []
        for ph_id in ph_id_list:
            pf_kv, _ = self._resolve_pf_for_ph(
                self.node_id,
                ph_id,
                prefix_store=prefix_store,
                prefix_kv_list=prefix_kv_list,
            )
            base_pf_list.append(pf_kv)
        return list(base_ph_list), base_pf_list

    def _sync_dense_anchor_to_pool(
        self,
        *,
        request_uid: str,
        message: str,
        ph_id_list: List[str],
    ) -> None:
        """Persist dense materialised deltas under topology-anchored keys."""
        from sidecar.stores.prefix_spans import normalize_placeholder_info
        from sidecar.stores.topology_anchor import delta_key_from_ph_rec, serialize_anchor_key

        state = self.get_request_state(request_uid)
        prefix_store = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        ph_info = normalize_placeholder_info(prefix_store.get("placeholder_info"))
        static_hash = str(prefix_store.get("static_template_hash") or "")
        topo = str(prefix_store.get("topology_id") or "")
        stores = self._get_store_registry()
        delta_ph = f"{self.node_id}_ph_key_delta"
        delta_pf = f"{self.node_id}_pf_key_delta"

        for ph_id in ph_id_list:
            ph_rec = ph_info.get(str(ph_id))
            if not isinstance(ph_rec, dict):
                continue
            entry = (state.anchors.get(ph_id) or {}).get(message)
            if not isinstance(entry, dict) or delta_ph not in entry:
                continue
            content_hash = self._upstream_agent_content_hash(str(ph_id), message) or ""
            delta_key = delta_key_from_ph_rec(
                ph_id=str(ph_id),
                ph_rec=ph_rec,
                static_template_hash=static_hash,
                topology_id=topo,
                content_hash=content_hash,
            )
            pf_len = self._prefix_segment_len_for_placeholder(str(ph_id))
            stores.agent_anchors.put(
                node_id=str(self.node_id),
                message_key=str(message),
                ph_id=str(ph_id),
                static_template_hash=static_hash,
                upstream_hash=str(message),
                ph_delta=entry.get(delta_ph),
                ph_value_delta=entry.get(f"{self.node_id}_ph_value_delta"),
                pf_delta=entry.get(delta_pf),
                pf_value_delta=entry.get(f"{self.node_id}_pf_value_delta"),
                pf_segment_len=pf_len,
                delta_key=delta_key,
            )
            entry["anchor_topology_key"] = serialize_anchor_key(delta_key)
            global_entry = (KVCOMMEngine.anchors.get(str(ph_id)) or {}).get(message)
            if isinstance(global_entry, dict):
                global_entry["anchor_topology_key"] = serialize_anchor_key(delta_key)

    async def _ensure_consumer_upstream_agent_slots(
        self,
        message: str,
        meta: List[Dict[str, Any]],
        *,
        tail_only_ph_ids: set[str] | None = None,
    ) -> None:
        """Materialize consumer-contextual KV for upstream agent placeholders."""
        from sidecar.stores.hashing import sha256_text

        stores = self._get_store_registry()
        bucket = LLMChat._shared_kv_cache_memory.get(str(self.node_id)) or {}
        materialized = bucket.setdefault("_upstream_materialized", {}).setdefault(str(message), set())
        tasks: list[tuple] = []

        for m in meta:
            ph_id = str(m["ph_id"])
            if tail_only_ph_ids is not None and ph_id not in tail_only_ph_ids and ph_id in materialized:
                continue
            upstream_idx = self._upstream_agent_index(ph_id)
            if upstream_idx is None:
                continue
            try:
                node_idx = int(self.node_id)
            except (TypeError, ValueError):
                continue
            if upstream_idx >= node_idx:
                continue

            ph_cache_ids = m.get("ph_cache_ids")
            if not isinstance(ph_cache_ids, dict) or ph_cache_ids.get("input_ids") is None:
                continue
            text = self.tokenizer.decode(ph_cache_ids["input_ids"][0], skip_special_tokens=True)
            content_hash = sha256_text(str(text).strip() or " ")
            existing = stores.upstream_agent_slots.get_consumer(
                str(self.node_id),
                str(message),
                ph_id,
                content_hash,
            )
            if existing is not None:
                continue

            drop_num = int(m.get("drop_num") or 0)
            tasks.append(
                (
                    str(self.node_id),
                    str(message),
                    ph_id,
                    int(m["start"]),
                    ph_cache_ids,
                    content_hash,
                    str(upstream_idx),
                    drop_num,
                )
            )

        if not tasks:
            return

        def _run_batch():
            for args in tasks:
                ok = self._materialize_consumer_upstream_slot_sync(
                    consumer_node_id=args[0],
                    message_key=args[1],
                    ph_id=args[2],
                    slot_token_start=args[3],
                    ph_token_ids=args[4],
                    content_hash=args[5],
                    upstream_node_id=args[6],
                    drop_num=args[7],
                )
                if ok:
                    materialized.add(args[2])

        await asyncio.to_thread(_run_batch)

    def _refresh_meta_from_upstream_slots(
        self,
        message: str,
        meta: List[Dict[str, Any]],
    ) -> None:
        """Prefer consumer-contextual KV in merge meta when materialized."""
        cum_offset = 0
        ph_cum_len = 0
        for m in meta:
            slot = self.resolve_upstream_agent_slot(str(m["ph_id"]), message)
            if slot is not None and slot.materialization == "consumer_contextual":
                m["ph_cache"] = slot.absolute_kv
                m["ph_cache_ids"] = slot.token_ids
                m["drop_num"] = int(slot.drop_num)
            ph_cache = m["ph_cache"]
            drop_num = int(m.get("drop_num") or 0)
            real_len = int(ph_cache._seen_tokens - drop_num)
            templ_len = int(m["end"]) - int(m["start"])
            delta_len = real_len - templ_len
            m["delta"] = delta_len
            m["offset_before"] = cum_offset
            m["offset_after"] = cum_offset + delta_len
            m["cum_len"] = ph_cum_len
            cum_offset += delta_len
            ph_cum_len += real_len

    async def append_prefix_segment(
        self,
        node_id: str,
        segment_template: str,
        *,
        system_prompt: str | None = None,
        expected_user_template: str | None = None,
    ) -> None:
        """Append a templated segment (with turn placeholders) to an existing prefix."""
        await self.append_prefix_segment_incremental(
            node_id,
            segment_template,
            system_prompt=system_prompt,
            expected_user_template=expected_user_template,
        )

    async def materialize_turn_placeholders(
        self,
        node_id: str,
        message_key: str,
        turn_content: Dict[str, str],
    ) -> None:
        """Materialize KV caches for completed assistant/tool turn placeholders."""
        if not turn_content:
            return

        from sidecar.stores.hashing import sha256_text
        from sidecar.stores.turn_slot_registry import TurnPhSlot

        stores = self._get_store_registry()
        mem = LLMChat._shared_kv_cache_memory.setdefault(node_id, {})

        for ph_id, content in turn_content.items():
            if not ph_id.startswith("turn_"):
                continue
            content_str = str(content).strip() or " "
            content_hash = sha256_text(content_str)
            turn_index = 0
            parts = ph_id.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                turn_index = int(parts[1])

            existing_slot = stores.turn_slots.get(node_id, message_key, ph_id)
            if existing_slot is not None and existing_slot.content_hash == content_hash:
                continue

            is_tool = ph_id.endswith("_tool")
            if is_tool:
                lookup = stores.tool_semantic.lookup(content_str, content_hash=content_hash)
                if lookup.hit and lookup.kv_ref:
                    tool_entry = stores.tool_kv.get(lookup.kv_ref)
                else:
                    tool_entry = stores.tool_kv.get_or_create(
                        content_str,
                        self._forward_text_to_kv_sync,
                    )
                    stores.tool_semantic.upsert(
                        query=content_str,
                        kv_ref=tool_entry.kv_ref,
                        content_hash=tool_entry.content_hash,
                        ph_id_hint=ph_id,
                        token_len=tool_entry.token_len,
                    )
                stores.turn_slots.put(
                    TurnPhSlot(
                        node_id=str(node_id),
                        message_key=str(message_key),
                        ph_id=str(ph_id),
                        slot_kind="tool",
                        content_hash=content_hash,
                        kv_ref=tool_entry.kv_ref,
                        token_ids=tool_entry.token_ids,
                        drop_num=0,
                        turn_index=turn_index,
                    )
                )
                continue

            kv_cache, token_ids = await asyncio.to_thread(self._forward_text_to_kv_sync, content_str)
            stores.turn_slots.put(
                TurnPhSlot(
                    node_id=str(node_id),
                    message_key=str(message_key),
                    ph_id=str(ph_id),
                    slot_kind="assistant",
                    content_hash=content_hash,
                    absolute_kv=kv_cache,
                    token_ids=token_ids,
                    drop_num=0,
                    turn_index=turn_index,
                )
            )

    def _forward_text_to_kv_sync(self, text: str) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run a short HF prefill for arbitrary text (turn/upstream response KV)."""
        content_str = str(text).strip() or " "
        token_ids = self.tokenizer(
            content_str,
            add_special_tokens=False,
            return_tensors="pt",
        )
        token_ids = {
            key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in token_ids.items()
        }
        token_ids["attention_mask"] = torch.ones_like(token_ids["input_ids"]).to(self.model.device)
        with torch.no_grad():
            with LLMChat._model_lock:
                output = self.model.generate(
                    **token_ids,
                    use_cache=True,
                    max_length=token_ids["input_ids"].shape[-1] + 1,
                    return_dict_in_generate=True,
                    return_legacy_cache=False,
                )
        kv_cache = output.past_key_values.slice_(
            start=0,
            end=token_ids["input_ids"].shape[-1],
        )
        return kv_cache, token_ids

    @staticmethod
    def _pretrained_kwargs(model_name: str) -> dict[str, Any]:
        """Use local_files_only when model_name points at an on-disk snapshot."""
        expanded = os.path.expanduser(model_name)
        if os.path.isdir(expanded):
            return {"local_files_only": True}
        return {}

    @staticmethod
    def _resolve_hf_dtype(model_name: str) -> torch.dtype:
        """Pick load dtype: bf16 for Qwen (matches checkpoint), fp16 for Llama."""
        name = model_name.lower()
        if "llama" in name:
            return torch.float16
        if "qwen" in name:
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _parse_gpu_pool(device: str) -> list[int]:
        if not device or device in ("auto", "balanced", "balanced_low_0", "sequential"):
            if not torch.cuda.is_available():
                return []
            return list(range(torch.cuda.device_count()))
        if "," in device:
            return [int(part.strip()) for part in device.split(",") if part.strip().isdigit()]
        if device.isdigit():
            return [int(device)]
        return []

    @staticmethod
    def _estimate_model_weight_gib(model_name: str) -> float:
        """Estimate on-disk weight bytes for device_map planning."""
        expanded = os.path.expanduser(model_name)
        if os.path.isdir(expanded):
            index_path = Path(expanded) / "model.safetensors.index.json"
            if index_path.exists():
                try:
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                    total = data.get("metadata", {}).get("total_size")
                    if total:
                        return float(total) / (1024**3)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            shard_bytes = sum(
                shard.stat().st_size for shard in Path(expanded).glob("*.safetensors")
            )
            if shard_bytes:
                return shard_bytes / (1024**3)
        return 64.0

    @staticmethod
    def _gpu_total_gib(gpu_id: int) -> float:
        return torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)

    @staticmethod
    def _gpu_free_gib(gpu_id: int) -> float:
        free, _ = torch.cuda.mem_get_info(gpu_id)
        return free / (1024**3)

    @staticmethod
    def _auto_plan_multi_gpu(model_name: str, gpu_pool: list[int]) -> dict[str, Any]:
        """Pick minimum GPUs from pool and cap per-GPU weight bytes to leave KV headroom."""
        if not gpu_pool:
            raise RuntimeError("KVCOMM_HF_DEVICE pool is empty and no CUDA devices are visible.")

        weight_gib = LLMChat._estimate_model_weight_gib(model_name)
        headroom_gib = float(os.environ.get("KVCOMM_HF_INFERENCE_HEADROOM_GIB", "12"))
        weight_margin_gib = float(os.environ.get("KVCOMM_HF_WEIGHT_MARGIN_GIB", "2"))
        dtype = LLMChat._resolve_hf_dtype(model_name)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for HF KVCOMM engine.")

        gpu_stats: list[dict[str, float | int]] = []
        for gpu_id in gpu_pool:
            total_gib = LLMChat._gpu_total_gib(gpu_id)
            free_gib = LLMChat._gpu_free_gib(gpu_id)
            # Explicit pool = user-designated HF cards; plan on total VRAM, not snapshot free.
            effective_total = total_gib
            if free_gib < total_gib * 0.20:
                logger.warning(
                    "GPU {} only {:.1f}GiB free of {:.1f}GiB total — ensure it is dedicated to HF sidecar.",
                    gpu_id,
                    free_gib,
                    total_gib,
                )
            weight_budget = max(1.0, effective_total - headroom_gib - weight_margin_gib)
            gpu_stats.append(
                {
                    "id": gpu_id,
                    "total_gib": total_gib,
                    "free_gib": free_gib,
                    "weight_budget_gib": weight_budget,
                }
            )

        selected: list[int] = []
        cumulative_budget = 0.0
        for stat in gpu_stats:
            selected.append(int(stat["id"]))
            cumulative_budget += float(stat["weight_budget_gib"])
            if cumulative_budget >= weight_gib * 1.01:
                break

        if cumulative_budget < weight_gib:
            raise RuntimeError(
                f"HF model needs ~{weight_gib:.1f}GiB weights (+{headroom_gib:.0f}GiB/GPU KV headroom). "
                f"Pool {gpu_pool} only budgets ~{cumulative_budget:.1f}GiB across "
                f"{len(gpu_pool)} GPU(s). Add more GPUs to KVCOMM_HF_DEVICE or free VRAM."
            )

        weight_share_gib = weight_gib / len(selected)
        max_memory: dict[int, str] = {}
        for gpu_id in selected:
            stat = next(item for item in gpu_stats if item["id"] == gpu_id)
            cap_gib = min(
                weight_share_gib + weight_margin_gib,
                float(stat["total_gib"]) * 0.90,
            )
            max_memory[gpu_id] = f"{max(1, int(cap_gib))}GiB"

        return {
            "device_map": "auto",
            "max_memory": max_memory,
            "torch_dtype": dtype,
            "_selected_gpus": selected,
            "_weight_gib": round(weight_gib, 2),
            "_headroom_gib": headroom_gib,
        }

    @staticmethod
    def _resolve_hf_load_kwargs(model_name: str) -> dict[str, Any]:
        """Build from_pretrained kwargs with automatic multi-GPU sharding.

        KVCOMM_HF_DEVICE: comma-separated **available GPU pool** (e.g. 2,3,4,5).
          The planner auto-picks the minimum subset that fits weights + KV headroom.
        KVCOMM_HF_INFERENCE_HEADROOM_GIB: KV/activation reserve per GPU (default 12).
        KVCOMM_HF_WEIGHT_MARGIN_GIB: extra weight cap margin per GPU (default 2).
        """
        device = os.environ.get("KVCOMM_HF_DEVICE", "").strip()
        dtype = LLMChat._resolve_hf_dtype(model_name)

        if device in ("auto", "balanced", "balanced_low_0", "sequential"):
            return {"device_map": device, "torch_dtype": dtype}

        gpu_pool = LLMChat._parse_gpu_pool(device)
        if not gpu_pool:
            return {"device_map": device or "cuda:0", "torch_dtype": dtype}

        weight_gib = LLMChat._estimate_model_weight_gib(model_name)
        headroom_gib = float(os.environ.get("KVCOMM_HF_INFERENCE_HEADROOM_GIB", "12"))
        weight_margin_gib = float(os.environ.get("KVCOMM_HF_WEIGHT_MARGIN_GIB", "2"))

        if len(gpu_pool) == 1:
            gpu_id = gpu_pool[0]
            if torch.cuda.is_available():
                total_gib = LLMChat._gpu_total_gib(gpu_id)
                effective = total_gib
                if weight_gib + headroom_gib + weight_margin_gib > effective:
                    needed = max(
                        2,
                        int(
                            (weight_gib + headroom_gib + weight_margin_gib - 1)
                            // max(1.0, effective - headroom_gib - weight_margin_gib)
                        )
                        + 1,
                    )
                    raise RuntimeError(
                        f"GPU {gpu_id} has ~{effective:.1f}GiB usable but model needs "
                        f"~{weight_gib:.1f}GiB weights + {headroom_gib:.0f}GiB KV headroom. "
                        f"Use at least {needed} GPUs in KVCOMM_HF_DEVICE."
                    )
            cap = int(min(weight_gib + weight_margin_gib, LLMChat._gpu_total_gib(gpu_id) * 0.90))
            return {
                "device_map": f"cuda:{gpu_id}",
                "max_memory": {gpu_id: f"{max(1, cap)}GiB"},
                "torch_dtype": dtype,
                "_selected_gpus": [gpu_id],
                "_weight_gib": round(weight_gib, 2),
            }

        plan = LLMChat._auto_plan_multi_gpu(model_name, gpu_pool)
        manual_cap = os.environ.get("KVCOMM_HF_MAX_MEMORY", "").strip()
        if manual_cap and plan.get("max_memory"):
            plan["max_memory"] = {
                gpu_id: manual_cap for gpu_id in plan["max_memory"]
            }
        return plan

    @staticmethod
    def _pretrained_load_kwargs(plan: dict[str, Any]) -> dict[str, Any]:
        """Strip planner metadata before passing kwargs to from_pretrained."""
        return {key: value for key, value in plan.items() if not str(key).startswith("_")}

    @classmethod
    def configured_gpu_pool(cls) -> list[int]:
        """Physical GPU ids configured for HF inference (never all visible devices)."""
        device = os.environ.get("KVCOMM_HF_DEVICE", "").strip()
        pool = cls._parse_gpu_pool(device)
        if pool:
            return pool
        try:
            plan = cls._resolve_hf_load_kwargs(
                os.environ.get("KVCOMM_HF_MODEL", "") or os.environ.get("KVCOMM_HF_MODEL_PATH", "")
            )
            selected = plan.get("_selected_gpus")
            if isinstance(selected, list):
                return [int(gpu_id) for gpu_id in selected]
        except Exception:
            pass
        return []

    @classmethod
    def _ensure_shared_kv_memory(cls) -> dict:
        """Return shared KV store dict; never leave the class attribute as None."""
        if cls._shared_kv_cache_memory is None:
            cls._shared_kv_cache_memory = {}
        return cls._shared_kv_cache_memory

    @classmethod
    def _clear_shared_kv_memory(cls) -> None:
        cls._ensure_shared_kv_memory().clear()

    @classmethod
    def _dispose_hf_model(cls, model: Any) -> None:
        if model is None:
            return
        try:
            from accelerate.hooks import remove_hook_from_module
        except ImportError:
            remove_hook_from_module = None
        if remove_hook_from_module is not None:
            for module in list(model.modules()):
                try:
                    remove_hook_from_module(module)
                except Exception:
                    pass
        try:
            model.to("cpu")
        except Exception:
            pass
        del model

    @classmethod
    def _flush_cuda_pool(cls, gpu_pool: list[int]) -> None:
        if not torch.cuda.is_available():
            return
        if not gpu_pool:
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            return
        for gpu_id in gpu_pool:
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass

    @classmethod
    def release_shared_resources(cls) -> bool:
        """Unload shared HF weights/tokenizer and free GPU memory."""
        import gc

        gpu_pool = cls.configured_gpu_pool()
        model = None
        tokenizer = None
        released = False
        with cls._model_lock:
            if (
                cls._shared_model is None
                and cls._shared_tokenizer is None
                and not cls._shared_kv_cache_memory
            ):
                return False
            model = cls._shared_model
            tokenizer = cls._shared_tokenizer
            cls._shared_model = None
            cls._shared_tokenizer = None
            cls._clear_shared_kv_memory()
            cls._initialization = {}
            released = True
        if cls._THREAD_POOL is not None:
            try:
                cls._THREAD_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            cls._THREAD_POOL = None
            cls._THREAD_POOL_WORKERS = None
        cls._dispose_hf_model(model)
        if tokenizer is not None:
            del tokenizer
        cls._request_states.clear()
        cls._active_requests.clear()
        cls._staged_commits.clear()
        gc.collect()
        cls._flush_cuda_pool(gpu_pool)
        return released

    @staticmethod
    def describe_hf_load_plan(model_name: str | None = None) -> str:
        """Human-readable summary for /health and logs."""
        model_name = model_name or os.environ.get("KVCOMM_HF_MODEL", "")
        plan = LLMChat._resolve_hf_load_kwargs(model_name)
        device_map = plan.get("device_map")
        dtype = plan.get("torch_dtype")
        max_memory = plan.get("max_memory")
        selected = plan.get("_selected_gpus")
        weight_gib = plan.get("_weight_gib")
        headroom = plan.get("_headroom_gib")
        parts = [f"device_map={device_map}", f"dtype={dtype}"]
        if selected:
            parts.append(f"selected_gpus={selected}")
        if weight_gib is not None:
            parts.append(f"weight_gib={weight_gib}")
        if headroom is not None:
            parts.append(f"kv_headroom_gib={headroom}")
        if max_memory:
            parts.append(f"max_memory={max_memory}")
        return " ".join(parts)

    def _initialize_shared_resources(self):
        """Lazy-load shared tokenizer/model and shared KV memory storage."""
        with LLMChat._model_lock:
            if LLMChat._shared_model is None:
                local_kwargs = self._pretrained_kwargs(self.model_name)
                load_plan = self._resolve_hf_load_kwargs(self.model_name)
                load_kwargs = self._pretrained_load_kwargs(load_plan)
                LLMChat._shared_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, **local_kwargs
                )
                LLMChat._shared_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **local_kwargs,
                    **load_kwargs,
                )
                logger.info(
                    "Model {} loaded ({}) and shared across instances.",
                    self.model_name,
                    self.describe_hf_load_plan(self.model_name),
                )
            LLMChat._ensure_shared_kv_memory()

    def locate_placeholder(self, original_text, return_segments=False):
        """Locate placeholder token spans in a templated prompt.

        Args:
            original_text: Templated prompt with placeholders such as
                "{agent_2_current}" or "{user_question}".
            return_segments: Whether to also return segment encodings.

        Returns:
            placeholder_info: Mapping placeholder -> [start, end] token indices.
            encoding or (encoding, segments): Tokenized input and optional segments.
        """

        placeholder_pattern = (
            r'\{((?:agent|condition)_\w+_(?:current|history)|user_question|turn_\d+_(?:assistant|tool))\}'
        )

        matches = list(re.finditer(placeholder_pattern, original_text))

        last_pos = 0
        segments = []
        placeholder_info = {}
        token_num = 0
        idx_count = 0
        for m in matches:
            start, end = m.span()
            placeholder_inner = m.group(1)
            if last_pos < start:
                txt = original_text[last_pos:start]
                token_id = self.tokenizer(txt, add_special_tokens=False)['input_ids']
                encoding = {}
                encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
                encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
                # Keep whitespace-only spans (e.g. "\n\n" between turn placeholders);
                # they are explicit pf text spans linked via pf_span_id in placeholder_info.
                if txt:
                    segments.append(("text", txt, encoding, token_num, token_num + len(token_id)))
                    idx_count += 1
                token_num += len(token_id)
            token_id = self.tokenizer(f'{ {placeholder_inner}} ', add_special_tokens=False)['input_ids']
            encoding = {}
            encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
            encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
            segments.append(("placeholder", placeholder_inner, encoding, token_num, token_num + len(token_id)))
            placeholder_info[placeholder_inner] = [token_num, token_num + len(token_id)]
            token_num += len(token_id)
            idx_count += 1
            last_pos = end

        txt = original_text[last_pos:]
        token_id = self.tokenizer(txt, add_special_tokens=False)['input_ids']
        encoding = {}
        encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
        encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
        if txt:
            segments.append(("text", txt, encoding, token_num, token_num + len(token_id)))
            token_num += len(token_id)

        segments.sort(key=lambda x: x[-1])
        token_ids = torch.cat([sublist[2]['input_ids'] for sublist in segments], dim=1)
        encoding = {}
        encoding['input_ids'] = token_ids
        encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)

        placeholder_info = dict(sorted(placeholder_info.items(), key=lambda x: x[1][0], reverse=True))
        if return_segments:
            return placeholder_info, encoding, segments
        return placeholder_info, encoding

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        pass

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_cache: Optional[bool] = False,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        async with self.lock:
            if max_tokens is None:
                max_tokens = self.DEFAULT_MAX_TOKENS
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE
            inputs, prompt_text, prompt_length = self._build_chat_inputs(messages)
            safe_prompt_text = _escape_loguru_markup(prompt_text)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT]</blue> Agent {} Role {} Prompt:\n{}",
                self.node_id,
                self.role,
                safe_prompt_text,
            )
            generation_kwargs = {
                "do_sample": False,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_dict_in_generate": True,
                "return_legacy_cache": False,
                "use_cache": True,
            }
            ttft_tracer = _TTFTTracer(prompt_length)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
            ttft_tracer.reset(prompt_length)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outputs = self.model.generate(**inputs, **generation_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_to_end_latency = perf_counter() - ttft_tracer.start_time
            if ttft_tracer.ttft is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_value = 0.0
            else:
                ttft_value = ttft_tracer.ttft
            generated_sequence = outputs.sequences[:, prompt_length:]
            response_message = _sanitize_chat_template_leaks(
                self.tokenizer.decode(
                    generated_sequence[0], skip_special_tokens=True
                ).strip()
            )
            safe_response_message = _escape_loguru_markup(response_message)
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE]</blue> Agent {} Role {} Response:\n{}",
                self.node_id,
                self.role,
                safe_response_message,
            )
            metadata: Dict[str, Any] = {}
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if agent_name:
                metadata["agent_name"] = agent_name
            if agent_role:
                metadata["agent_role"] = agent_role
            metadata["kvcomm_latency"] = 0.0
            metadata["first_token_decode"] = ttft_value
            metadata["generation_ttft"] = ttft_value
            metadata["others_ttft"] = max(0.0, ttft_value - metadata["kvcomm_latency"] - metadata["first_token_decode"])
            metadata["others_e2e"] = max(0.0, end_to_end_latency - ttft_value)
            metadata["others_latency"] = metadata["others_e2e"]
            if return_cache:
                metadata["kv_cache"] = outputs.past_key_values
            return GenerationResult(
                text=response_message,
                mode="default",
                ttft=ttft_value,
                metadata=metadata,
            )

    @retry(wait=wait_random_exponential(max=1000), stop=stop_after_attempt(1))
    async def generate_with_dense_prefill(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_anchor_num: Optional[int] = 20,
        window_length: Optional[int] = 5,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate with dense prefix prefill and optional anchor update."""
        return await self.agen_kvcomm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            mode="dense_prefill",
            max_anchor_num=max_anchor_num,
            window_length=window_length,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir,
            **kwargs,
        )

    @retry(wait=wait_random_exponential(max=1000), stop=stop_after_attempt(1))
    async def generate_with_kv_reuse(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate by reusing existing prefix KV (fast path)."""
        test_time = kwargs.get("test_time", False)
        if test_time:
            return await self.agen_kvcomm_time_test(
                messages=messages,
                max_tokens=max_tokens,
                min_tokens=max_tokens,
                temperature=temperature,
                request_uid=request_uid,
                mode="kv_reuse",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=output_dir,
            )
        return await self.agen_kvcomm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            mode="kv_reuse",
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir,
            **kwargs,
        )

    async def agen_kvcomm(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        max_anchor_num: int = 20,
        window_length: int = 5,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        on_token: Optional[Any] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Core KV-aware generation entry.

        Builds merged prefix KV and token ids from stored segments and per-request
        placeholder caches, then runs generation either with:
        - dense_prefill: compute fresh prefix KV and optionally set anchors
        - kv_reuse: reuse existing prefix KV and inject as past_key_values
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm.")
        state = self.kv_engine.resolve_request_state(request_uid)
        preprocess_start = perf_counter()

        if isinstance(messages, List):
            message = messages[0]
        else:
            message = messages

        prefix_store = self._shared_kv_cache_memory.get(self.node_id) or {}
        if not isinstance(prefix_store, dict):
            raise RuntimeError(
                f"Invalid shared KV memory for node '{self.node_id}' (expected dict, got {type(prefix_store).__name__})."
            )
        prefix_kv_list: List[DynamicCache] = prefix_store.get("prefix", [])
        prefix_token_ids: List[Dict[str, torch.Tensor]] = prefix_store.get("token_ids", [])
        placeholder_info_map = prefix_store.get("placeholder_info")
        if not prefix_kv_list:
            raise RuntimeError(
                "No prefix KV found in shared memory. Make sure you've called prepare_prefix_kv_segments or init_shared_placeholder_prefix_kv."
            )
        if placeholder_info_map is None:
            raise RuntimeError("placeholder_info missing in shared KV cache memory.")

        from sidecar.stores.prefix_spans import normalize_placeholder_info, ordered_placeholders

        merged_prefix_kv = prefix_kv_list[0].copy()
        merged_prefix_token_ids = prefix_token_ids[0].copy()

        placeholder_info_norm = normalize_placeholder_info(placeholder_info_map)

        meta: List[Dict[str, Any]] = []
        ph_id_list: List[str] = []
        cum_offset = 0
        ph_cum_len = 0
        for idx, (ph_id, ph_rec) in enumerate(ordered_placeholders(placeholder_info_norm)):
            start = int(ph_rec["start"])
            end = int(ph_rec["end"])
            pf_kv, pf_token_id = self._resolve_pf_for_ph(
                self.node_id,
                ph_id,
                prefix_store=prefix_store,
                prefix_kv_list=prefix_kv_list,
                prefix_token_ids=prefix_token_ids,
            )
            ph_cache, ph_cache_ids, drop_num = self.kv_engine.fetch_shared_cache(ph_id, message)
            real_len = ph_cache._seen_tokens - drop_num
            templ_len = end - start
            delta_len = real_len - templ_len
            meta.append(
                {
                    "idx": idx,
                    "ph_id": ph_id,
                    "start": start,
                    "end": end,
                    "drop_num": drop_num,
                    "delta": delta_len,
                    "offset_before": cum_offset,
                    "offset_after": cum_offset + delta_len,
                    "ph_cache": ph_cache,
                    "ph_cache_ids": ph_cache_ids,
                    "pf_kv": pf_kv,
                    "pf_ids": pf_token_id,
                    "cum_len": ph_cum_len,
                }
            )
            cum_offset += delta_len
            ph_cum_len += real_len
            ph_id_list.append(ph_id)

        if mode == "kv_reuse":
            from sidecar.stores.topology_anchor import new_tail_placeholder_ids

            prev_ph = (prefix_store.get("_prev_placeholder_info") or {}) if isinstance(prefix_store, dict) else {}
            tail_ph_ids = new_tail_placeholder_ids(prev_ph, placeholder_info_norm)
            await self._ensure_consumer_upstream_agent_slots(
                message,
                meta,
                tail_only_ph_ids=tail_ph_ids if tail_ph_ids else None,
            )
            self._refresh_meta_from_upstream_slots(message, meta)

        reuse_kv_segments: List[Dict[str, Any]] = []
        if mode == "kv_reuse":
            for m in meta:
                ph_cache_ids = m.get("ph_cache_ids")
                real_len = int(m["ph_cache"]._seen_tokens - m["drop_num"])
                seg_text = ""
                if isinstance(ph_cache_ids, dict) and "input_ids" in ph_cache_ids:
                    seg_text = self.tokenizer.decode(
                        ph_cache_ids["input_ids"][0],
                        skip_special_tokens=True,
                    )
                reuse_kv_segments.append(
                    {
                        "ph_id": m["ph_id"],
                        "tokens": real_len,
                        "text": seg_text[:500],
                    }
                )

        initial_mode = mode
        blend_fallback = False
        if mode == "dense_prefill":
            tasks = [(message, m) for m in meta]
            results = list(
                self._map_in_pool(self.kv_engine.process_anchor, tasks, timeout=30)
            )
        elif mode == "kv_reuse":
            anchors_for_node = state.anchors
            tasks = [
                (
                    request_uid,
                    message,
                    m,
                    self._kv_reuse_anchors_for_ph(m["ph_id"], message, anchors_for_node),
                )
                for m in meta
            ]
            results = list(self._map_in_pool(self.kv_engine.update_kv_cache_segment, tasks, timeout=30))
            blend_failed = [m["ph_id"] for m, result in zip(meta, results) if not result[3]]
            if blend_failed:
                blend_fallback = True
                self.purge_incompatible_anchor_deltas(request_uid, message, blend_failed)
                logger.debug(
                    "kv_reuse anchor blend failed node_id={} placeholders={} -> dense_prefill rematerialize",
                    self.node_id,
                    blend_failed,
                )
                mode = "dense_prefill"
                results = list(
                    self._map_in_pool(
                        self.kv_engine.process_anchor,
                        [(message, m) for m in meta],
                        timeout=30,
                    )
                )
        else:
            raise ValueError(f"Unsupported mode '{mode}' for agen_kvcomm.")

        results_sorted = sorted(results, key=lambda x: x[0])

        placeholder_indices: Dict[str, Tuple[int, int]] = {}
        for m in meta:
            start = m["start"] + m["offset_before"]
            placeholder_indices[m["ph_id"]] = (
                start,
                start + m["ph_cache"]._seen_tokens - m["drop_num"],
            )

        seg_cache_list = [r[1] for r in results_sorted]
        merged_prefix_kv.concat_(seg_cache_list)
        seg_ids_list = [r[2] for r in results_sorted]
        merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)

        cached_prefix_token_length, generation_prompt_length = _reconcile_prefix_kv_and_tokens(
            merged_prefix_kv,
            merged_prefix_token_ids,
        )

        if "position_ids" in merged_prefix_token_ids:
            merged_prefix_token_ids["position_ids"] = (
                torch.arange(generation_prompt_length).unsqueeze(0).to(self.model.device)
            )

        tool_injection_text = kwargs.get("tool_injection_text")
        if isinstance(tool_injection_text, str) and tool_injection_text.strip():
            suffix_tokens = self.tokenizer(
                tool_injection_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            suffix_tokens = {
                key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
                for key, value in suffix_tokens.items()
            }
            merged_prefix_token_ids = concat_(merged_prefix_token_ids, suffix_tokens)
            generation_prompt_length = int(merged_prefix_token_ids["input_ids"].shape[-1])
            if "position_ids" in merged_prefix_token_ids:
                merged_prefix_token_ids["position_ids"] = (
                    torch.arange(generation_prompt_length).unsqueeze(0).to(self.model.device)
                )

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "temperature": temperature,
            "return_legacy_cache": False,
            "return_dict_in_generate": True,
        }
        if isinstance(tool_injection_text, str) and tool_injection_text.strip():
            generation_kwargs["repetition_penalty"] = 1.08

        if mode == "kv_reuse":
            merged_prefix_kv = merged_prefix_kv.slice_(start=0, end=cached_prefix_token_length - 1)
            generation_kwargs["past_key_values"] = merged_prefix_kv

        ttft_tracer = _TTFTTracer(generation_prompt_length)
        ttft_tracer.reset(generation_prompt_length)
        stream_cb = None
        token_cb = on_token or kwargs.get("on_token")
        if token_cb is not None:
            stream_cb = _TokenStreamCallback(self.tokenizer, generation_prompt_length, token_cb)
            stream_cb.reset(generation_prompt_length)
        criteria: List[StoppingCriteria] = [ttft_tracer]
        if stream_cb is not None:
            criteria.append(stream_cb)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)
        preprocess_latency = max(0.0, perf_counter() - preprocess_start)

        def _run_model_generate():
            with LLMChat._model_lock:
                return self.model.generate(**merged_prefix_token_ids, **generation_kwargs)

        outputs = await asyncio.to_thread(_run_model_generate)
        if ttft_tracer.ttft is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_ttft = 0.0
        else:
            generation_ttft = ttft_tracer.ttft
        ttft_value = generation_ttft + preprocess_latency

        full_kv_cache = outputs.past_key_values

        if mode == "dense_prefill":
            base_cache = merged_prefix_kv
            real_cache = full_kv_cache.slice(start=0, end=cached_prefix_token_length)
            real_placeholder_cache, real_prefix_cache = real_cache.split_cache_by_placeholders(
                placeholder_indices
            )
            base_placeholder_cache, base_prefix_cache = self._resolve_base_caches_for_dense_anchor(
                ph_id_list,
                merged_prefix_kv,
                placeholder_indices,
            )
            self.kv_engine.set_anchor(
                request_uid,
                message,
                ph_id_list,
                real_placeholder_cache,
                real_prefix_cache,
                base_placeholder_cache,
                base_prefix_cache,
                max_anchor_num=max_anchor_num,
                window_length=window_length,
            )
            self._sync_dense_anchor_to_pool(
                request_uid=request_uid,
                message=message,
                ph_id_list=ph_id_list,
            )

        seq = outputs.sequences
        generated_ids = _trim_token_ids_at_eos(
            self.tokenizer,
            seq[0, generation_prompt_length:].unsqueeze(0),
        )[0]
        raw_response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        try:
            from sidecar.tool_bridge import sanitize_generation_text
        except ImportError:
            sanitize_generation_text = _sanitize_chat_template_leaks  # type: ignore[assignment]
        stored_text = sanitize_generation_text(raw_response_text)
        if not stored_text and raw_response_text.strip():
            stored_text = _sanitize_chat_template_leaks(raw_response_text)

        response_tokens = generated_ids.unsqueeze(0)
        response_kv_cache, token_dict = self._export_producer_contextual_response_kv(
            full_kv_cache=full_kv_cache,
            prefix_token_len=int(cached_prefix_token_length),
            response_token_ids=generated_ids,
            message=message,
            stored_text=stored_text or " ",
        )
        attn_len = response_tokens.size(1)
        response_mask = torch.ones(response_tokens.size(0), attn_len, device=self.model.device)

        mem = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        if not isinstance(mem, dict):
            raise RuntimeError(
                f"Invalid shared KV memory for node '{self.node_id}' while storing response KV."
            )
        resp = mem.setdefault("response", {})
        resp_ids = mem.setdefault("response_ids", {})
        resp_drop = mem.setdefault("response_drop_num", {})

        current_key = f"agent_{self.node_id}_current"
        current_bucket = state.anchor_dict.get(current_key) or {}
        had_prior_response_anchor = bool(
            current_bucket.get(message) if isinstance(current_bucket, dict) else None
        )
        anchor_bucket = state.anchors.setdefault(current_key, {})
        anchor_len_bucket = state.anchor_len_dict.setdefault(current_key, {})
        anchor_info_bucket = state.anchor_info_dict.setdefault(current_key, {})
        response_anchor_list = list(anchor_bucket.values())
        anchor_len_list = [
            anchor_len_bucket.get(kk, [0, 0])
            for kk in anchor_bucket.keys()
        ]
        anchor_active_list: List[int] = list(anchor_info_bucket.values())

        resp.setdefault(message, []).append(response_kv_cache)
        resp_ids.setdefault(message, []).append(
            {
                "input_ids": response_tokens,
                "attention_mask": response_mask,
            }
        )
        resp_drop.setdefault(message, []).append(0)

        accumulate_len = 0
        for key in state.anchor_len_dict.keys():
            bucket = state.anchor_len_dict.get(key) or {}
            if not isinstance(bucket, dict):
                continue
            length_entry = bucket.get(message, [0, 0])
            if isinstance(length_entry, (list, tuple)) and length_entry:
                accumulate_len += length_entry[0]

        prob, anchor_active_list = self.kv_engine.predict_as_anchor(
            response_kv_cache,
            anchor_kv_cache_list=response_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_active_list,
        )
        safe_message = _escape_loguru_markup(message)
        logger.opt(colors=True).debug(
            f"<magenta>Agent {self.node_id} Role {self.role} Message {safe_message} Response Anchor Prediction: {prob}</magenta>",
        )
        state.anchor_dict.setdefault(current_key, {})[message] = prob

        if not prob:
            global_bucket = state.global_anchor_info.setdefault(current_key, {})
            info_items = list(anchor_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                anchor_info_bucket[msg_key] = anchor_active_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_active_list[idx]

        response_message = (stored_text or raw_response_text).strip()
        prompt_preview = self.tokenizer.decode(
            merged_prefix_token_ids["input_ids"][0]
        )
        safe_prompt_preview = _escape_loguru_markup(prompt_preview)
        safe_response_message = _escape_loguru_markup(response_message)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:{mode}]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id,
            self.role,
            safe_prompt_preview,
            mode=mode,
        )
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:{mode}]</blue> Agent {} Role {} Response:\n{}",
            self.node_id,
            self.role,
            safe_response_message,
            mode=mode,
        )

        metadata: Dict[str, Any] = {
            "placeholder_ids": ph_id_list,
            "anchor_prediction": bool(prob),
            "anchor_pooled_tokens": (
                int(response_tokens.numel()) if prob and not had_prior_response_anchor else 0
            ),
            "blend_fallback": blend_fallback,
            "routed_mode": initial_mode,
        }
        if reuse_kv_segments:
            joined_reuse = " | ".join(
                segment["text"] for segment in reuse_kv_segments if segment.get("text")
            )
            compact_reuse = joined_reuse.replace("\n", " ").strip()
            if len(compact_reuse) > 500:
                compact_reuse = f"{compact_reuse[:497]}..."
            metadata["reuse_kv_segments"] = reuse_kv_segments
            metadata["reuse_kv_text"] = compact_reuse
        if isinstance(tool_injection_text, str) and tool_injection_text.strip():
            metadata["tool_injection_tokens"] = max(
                0,
                generation_prompt_length - cached_prefix_token_length,
            )
        metadata["preprocess_latency"] = preprocess_latency
        metadata["generation_ttft"] = generation_ttft
        metadata["kvcomm_latency"] = preprocess_latency
        metadata["first_token_decode"] = generation_ttft
        metadata["others_ttft"] = max(
            0.0,
            ttft_value - preprocess_latency - generation_ttft,
        )
        metadata["others_e2e"] = None
        metadata["others_latency"] = metadata["others_ttft"]
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        latency_record = {
            "timestamp": time.time(),
            "mode": mode,
            "ttft": float(ttft_value),
            "generation_ttft": float(generation_ttft),
            "preprocess_latency": float(preprocess_latency),
            "kvcomm_latency": float(preprocess_latency),
            "first_token_decode": float(generation_ttft),
            "others_ttft": float(max(0.0, ttft_value - preprocess_latency - generation_ttft)),
            "others_e2e": None,
            "others_latency": float(max(0.0, ttft_value - preprocess_latency - generation_ttft)),
            "request_uid": request_uid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "message": str(message) if message is not None else None,
            "placeholder_ids": ph_id_list,
            "anchor_prediction": bool(prob),
        }
        _append_latency_record(output_dir, latency_record)
        return GenerationResult(
            text=response_message,
            mode=mode,
            ttft=ttft_value,
            metadata=metadata,
        )

    async def agen_kvcomm_time_test(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        max_anchor_num: int = 20,
        window_length: int = 5,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Core KV-aware generation entry.

        Builds merged prefix KV and token ids from stored segments and per-request
        placeholder caches, then runs generation either with:
        - dense_prefill: compute fresh prefix KV and optionally set anchors
        - kv_reuse: reuse existing prefix KV and inject as past_key_values
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm.")
        min_tokens = max_tokens if min_tokens is None else min_tokens
        state = self.kv_engine.resolve_request_state(request_uid)
        preprocess_start = perf_counter() if mode == "kv_reuse" else None

        if isinstance(messages, List):
            message = messages[0]
        else:
            message = messages

        prefix_store = self._shared_kv_cache_memory.get(self.node_id) or {}
        if not isinstance(prefix_store, dict):
            raise RuntimeError(
                f"Invalid shared KV memory for node '{self.node_id}' (expected dict, got {type(prefix_store).__name__})."
            )
        prefix_kv_list: List[DynamicCache] = prefix_store.get("prefix", [])
        prefix_token_ids: List[Dict[str, torch.Tensor]] = prefix_store.get("token_ids", [])
        placeholder_info_map = prefix_store.get("placeholder_info")
        if not prefix_kv_list:
            raise RuntimeError(
                "No prefix KV found in shared memory. Make sure you've called prepare_prefix_kv_segments or init_shared_placeholder_prefix_kv."
            )
        if placeholder_info_map is None:
            raise RuntimeError("placeholder_info missing in shared KV cache memory.")

        from sidecar.stores.prefix_spans import normalize_placeholder_info, ordered_placeholders

        merged_prefix_kv = prefix_kv_list[0].copy()
        merged_prefix_token_ids = prefix_token_ids[0].copy()

        placeholder_info_norm = normalize_placeholder_info(placeholder_info_map)

        meta: List[Dict[str, Any]] = []
        ph_id_list: List[str] = []
        cum_offset = 0
        ph_cum_len = 0
        for idx, (ph_id, ph_rec) in enumerate(ordered_placeholders(placeholder_info_norm)):
            start = int(ph_rec["start"])
            end = int(ph_rec["end"])
            pf_kv, pf_token_id = self._resolve_pf_for_ph(
                self.node_id,
                ph_id,
                prefix_store=prefix_store,
                prefix_kv_list=prefix_kv_list,
                prefix_token_ids=prefix_token_ids,
            )
            ph_cache, ph_cache_ids, drop_num = self.kv_engine.fetch_shared_cache(ph_id, message)
            real_len = ph_cache._seen_tokens - drop_num
            templ_len = end - start
            delta_len = real_len - templ_len
            meta.append(
                {
                    "idx": idx,
                    "ph_id": ph_id,
                    "start": start,
                    "end": end,
                    "drop_num": drop_num,
                    "delta": delta_len,
                    "offset_before": cum_offset,
                    "offset_after": cum_offset + delta_len,
                    "ph_cache": ph_cache,
                    "ph_cache_ids": ph_cache_ids,
                    "pf_kv": pf_kv,
                    "pf_ids": pf_token_id,
                    "cum_len": ph_cum_len,
                }
            )
            cum_offset += delta_len
            ph_cum_len += real_len
            ph_id_list.append(ph_id)

        if mode == "kv_reuse":
            from sidecar.stores.topology_anchor import new_tail_placeholder_ids

            prev_ph = (prefix_store.get("_prev_placeholder_info") or {}) if isinstance(prefix_store, dict) else {}
            tail_ph_ids = new_tail_placeholder_ids(prev_ph, placeholder_info_norm)
            await self._ensure_consumer_upstream_agent_slots(
                message,
                meta,
                tail_only_ph_ids=tail_ph_ids if tail_ph_ids else None,
            )
            self._refresh_meta_from_upstream_slots(message, meta)

        reuse_kv_segments: List[Dict[str, Any]] = []
        if mode == "kv_reuse":
            for m in meta:
                ph_cache_ids = m.get("ph_cache_ids")
                real_len = int(m["ph_cache"]._seen_tokens - m["drop_num"])
                seg_text = ""
                if isinstance(ph_cache_ids, dict) and "input_ids" in ph_cache_ids:
                    seg_text = self.tokenizer.decode(
                        ph_cache_ids["input_ids"][0],
                        skip_special_tokens=True,
                    )
                reuse_kv_segments.append(
                    {
                        "ph_id": m["ph_id"],
                        "tokens": real_len,
                        "text": seg_text[:500],
                    }
                )

        initial_mode = mode
        blend_fallback = False
        if mode == "dense_prefill":
            tasks = [(message, m) for m in meta]
            results = list(
                self._map_in_pool(self.kv_engine.process_anchor, tasks, timeout=30)
            )
        elif mode == "kv_reuse":
            anchors_for_node = state.anchors
            tasks = [
                (
                    request_uid,
                    message,
                    m,
                    self._kv_reuse_anchors_for_ph(m["ph_id"], message, anchors_for_node),
                )
                for m in meta
            ]
            results = list(self._map_in_pool(self.kv_engine.update_kv_cache_segment, tasks, timeout=30))
            blend_failed = [m["ph_id"] for m, result in zip(meta, results) if not result[3]]
            if blend_failed:
                blend_fallback = True
                self.purge_incompatible_anchor_deltas(request_uid, message, blend_failed)
                logger.debug(
                    "kv_reuse anchor blend failed node_id={} placeholders={} -> dense_prefill rematerialize",
                    self.node_id,
                    blend_failed,
                )
                mode = "dense_prefill"
                results = list(
                    self._map_in_pool(
                        self.kv_engine.process_anchor,
                        [(message, m) for m in meta],
                        timeout=30,
                    )
                )
        else:
            raise ValueError(f"Unsupported mode '{mode}' for agen_kvcomm.")

        results_sorted = sorted(results, key=lambda x: x[0])

        placeholder_indices: Dict[str, Tuple[int, int]] = {}
        for m in meta:
            start = m["start"] + m["offset_before"]
            placeholder_indices[m["ph_id"]] = (
                start,
                start + m["ph_cache"]._seen_tokens - m["drop_num"],
            )

        seg_cache_list = [r[1] for r in results_sorted]
        merged_prefix_kv.concat_(seg_cache_list)
        seg_ids_list = [r[2] for r in results_sorted]
        merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)

        prefix_token_length, input_length = _reconcile_prefix_kv_and_tokens(
            merged_prefix_kv,
            merged_prefix_token_ids,
        )

        if "position_ids" in merged_prefix_token_ids:
            merged_prefix_token_ids["position_ids"] = (
                torch.arange(input_length).unsqueeze(0).to(self.model.device)
            )

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "min_new_tokens": min_tokens,
            "do_sample": False,
            "temperature": temperature,
            "return_legacy_cache": False,
            "return_dict_in_generate": True,
        }

        if mode == "kv_reuse":
            merged_prefix_kv = merged_prefix_kv.slice_(start=0, end=prefix_token_length - 1)
            generation_kwargs["past_key_values"] = merged_prefix_kv

        preprocess_latency = 0.0
        if preprocess_start is not None:
            torch.cuda.synchronize()
            preprocess_latency = max(0.0, perf_counter() - preprocess_start)
        torch.cuda.synchronize()
        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        outputs = self.model.generate(**merged_prefix_token_ids, **generation_kwargs)
        torch.cuda.synchronize()
        kvcomm_generation_ttft = ttft_tracer.ttft if ttft_tracer.ttft is not None else 0.0
        if mode == "kv_reuse" and preprocess_start is not None:
            kvcomm_end_to_end_latency = perf_counter() - ttft_tracer.start_time
            kvcomm_ttft_value = kvcomm_generation_ttft + preprocess_latency
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} KVCOMM E2E Latency: {kvcomm_end_to_end_latency:.4f}s TTFT: {kvcomm_ttft_value:.4f}s (Preprocess: {preprocess_latency:.4f}s)</green>",
            )
        full_kv_cache = outputs.past_key_values

        generation_kwargs.pop("past_key_values", None)
        torch.cuda.synchronize()
        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        _ = self.model.generate(**merged_prefix_token_ids, **generation_kwargs)
        torch.cuda.synchronize()
        dense_end_to_end_latency = perf_counter() - ttft_tracer.start_time
        dense_prefill_ttft = ttft_tracer.ttft if ttft_tracer.ttft is not None else 0.0
        logger.opt(colors=True).info(
            f"<cyan>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} Dense Prefill E2E Latency: {dense_end_to_end_latency:.4f}s TTFT: {dense_prefill_ttft:.4f}s</cyan>",
        )
        if mode == "kv_reuse" and preprocess_start is not None and kvcomm_ttft_value > 0:
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} KVCOMM is {dense_prefill_ttft / kvcomm_ttft_value:.2f}x faster than Dense Prefill in TTFT</green>",
            )
            ttft_value = kvcomm_ttft_value
        else:
            ttft_value = dense_prefill_ttft
        response_kv_cache = full_kv_cache.slice_(start=prefix_token_length)
        response_kv_cache = self.kv_engine.apply_rotary_pos_emb(
            response_kv_cache,
            offset=-prefix_token_length,
        )

        mem = LLMChat._shared_kv_cache_memory.get(self.node_id) or {}
        if not isinstance(mem, dict):
            raise RuntimeError(
                f"Invalid shared KV memory for node '{self.node_id}' while storing response KV."
            )
        resp = mem.setdefault("response", {})
        resp_ids = mem.setdefault("response_ids", {})
        resp_drop = mem.setdefault("response_drop_num", {})

        seq = outputs.sequences
        generated_ids = _trim_token_ids_at_eos(
            self.tokenizer,
            seq[0, prefix_token_length:].unsqueeze(0),
        )[0]
        response_tokens = generated_ids.unsqueeze(0)
        attn_len = response_tokens.size(1)
        response_mask = torch.ones(seq.size(0), attn_len, device=self.model.device)

        current_key = f"agent_{self.node_id}_current"
        current_bucket = state.anchor_dict.get(current_key) or {}
        had_prior_response_anchor = bool(
            current_bucket.get(message) if isinstance(current_bucket, dict) else None
        )
        anchor_bucket = state.anchors.setdefault(current_key, {})
        anchor_len_bucket = state.anchor_len_dict.setdefault(current_key, {})
        anchor_info_bucket = state.anchor_info_dict.setdefault(current_key, {})
        response_anchor_list = list(anchor_bucket.values())
        anchor_len_list = [
            anchor_len_bucket.get(kk, [0, 0])
            for kk in anchor_bucket.keys()
        ]
        anchor_active_list: List[int] = list(anchor_info_bucket.values())

        resp.setdefault(message, []).append(response_kv_cache)
        resp_ids.setdefault(message, []).append(
            {
                "input_ids": response_tokens,
                "attention_mask": response_mask,
            }
        )
        resp_drop.setdefault(message, []).append(0)

        accumulate_len = 0
        for key in state.anchor_len_dict.keys():
            bucket = state.anchor_len_dict.get(key) or {}
            if not isinstance(bucket, dict):
                continue
            length_entry = bucket.get(message, [0, 0])
            if isinstance(length_entry, (list, tuple)) and length_entry:
                accumulate_len += length_entry[0]

        prob, anchor_active_list = self.kv_engine.predict_as_anchor(
            response_kv_cache,
            anchor_kv_cache_list=response_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_active_list,
            test_time=True,
        )
        safe_message = _escape_loguru_markup(message)
        logger.opt(colors=True).debug(
            f"<magenta>Agent {self.node_id} Role {self.role} Message {safe_message} Response Anchor Prediction: {prob}</magenta>",
        )
        state.anchor_dict.setdefault(current_key, {})[message] = prob

        if not prob:
            global_bucket = state.global_anchor_info.setdefault(current_key, {})
            info_items = list(anchor_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                anchor_info_bucket[msg_key] = anchor_active_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_active_list[idx]

        response_message = _sanitize_chat_template_leaks(
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
        )
        prompt_preview = self.tokenizer.decode(
            merged_prefix_token_ids["input_ids"][0]
        )
        safe_prompt_preview = _escape_loguru_markup(prompt_preview)
        safe_response_message = _escape_loguru_markup(response_message)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:{mode}]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id,
            self.role,
            safe_prompt_preview,
            mode=mode,
        )
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:{mode}]</blue> Agent {} Role {} Response:\n{}",
            self.node_id,
            self.role,
            safe_response_message,
            mode=mode,
        )

        metadata: Dict[str, Any] = {
            "placeholder_ids": ph_id_list,
        }
        if preprocess_start is not None:
            metadata["preprocess_latency"] = preprocess_latency
            metadata["generation_ttft"] = ttft_value - preprocess_latency
            metadata["kvcomm_latency"] = preprocess_latency
            metadata["first_token_decode"] = metadata["generation_ttft"]
            metadata["others_ttft"] = max(
                0.0,
                ttft_value - metadata["kvcomm_latency"] - metadata["first_token_decode"],
            )
            if mode == "kv_reuse":
                metadata["others_e2e"] = max(
                    0.0,
                    kvcomm_end_to_end_latency - ttft_value,
                )
            else:
                metadata["others_e2e"] = max(
                    0.0,
                    dense_end_to_end_latency - ttft_value,
                )
            metadata["others_latency"] = metadata["others_e2e"]
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        if mode == "kv_reuse":
            latency_record = {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(ttft_value),
                "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "kvcomm_latency": float(metadata["kvcomm_latency"]) if "kvcomm_latency" in metadata else 0.0,
                "first_token_decode": float(metadata["first_token_decode"]) if "first_token_decode" in metadata else None,
                "others_ttft": float(metadata["others_ttft"]) if "others_ttft" in metadata else None,
                "others_e2e": float(metadata["others_e2e"]) if "others_e2e" in metadata else None,
                "others_latency": float(metadata["others_latency"]) if "others_latency" in metadata else None,
                "dense_prefill_ttft": float(dense_prefill_ttft),
                "kvcomm_end_to_end_latency": float(kvcomm_end_to_end_latency),
                "dense_end_to_end_latency": float(dense_end_to_end_latency),
                "ttft_ratio_dense_over_kvcomm": float(dense_prefill_ttft / ttft_value) if ttft_value > 0 else None,
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "message": str(message) if message is not None else None,
                "placeholder_ids": ph_id_list,
            }
        else:
            latency_record = {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(ttft_value),
                "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "kvcomm_latency": float(metadata["kvcomm_latency"]) if "kvcomm_latency" in metadata else 0.0,
                "first_token_decode": float(metadata["first_token_decode"]) if "first_token_decode" in metadata else float(dense_prefill_ttft),
                "others_ttft": float(metadata["others_ttft"]) if "others_ttft" in metadata else float(max(0.0, ttft_value - preprocess_latency - (metadata["first_token_decode"] if "first_token_decode" in metadata else dense_prefill_ttft))),
                "others_e2e": float(metadata["others_e2e"]) if "others_e2e" in metadata else float(max(0.0, dense_end_to_end_latency - ttft_value)),
                "others_latency": float(metadata["others_latency"]) if "others_latency" in metadata else float(max(0.0, dense_end_to_end_latency - ttft_value)),
                "dense_prefill_ttft": float(dense_prefill_ttft),
                "dense_end_to_end_latency": float(dense_end_to_end_latency),
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "message": str(message) if message is not None else None,
                "placeholder_ids": ph_id_list,
            }
        _append_latency_record(output_dir, latency_record)
        return GenerationResult(
            text=response_message,
            mode=mode,
            ttft=ttft_value,
            metadata=metadata,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['tokenizer']
        del state['lock']
        del state['_shared_kv_cache_memory']
        del state['_initialization']
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._shared_kv_cache_memory = LLMChat._shared_kv_cache_memory
        self._initialization = LLMChat._initialization
        self.lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = self.__getstate__()
        copied_state = copy.deepcopy(state, memo)
        node_id = copied_state.get('node_id', None)
        role = copied_state.get('role', None)
        if node_id is not None:
            if node_id in LLMChat._shared_kv_cache_memory:
                original_cache = LLMChat._shared_kv_cache_memory.get(node_id) or {}
                if not isinstance(original_cache, dict):
                    original_cache = {}
                LLMChat._shared_kv_cache_memory[node_id] = {
                    "prefix": original_cache.get("prefix"),
                    "placeholder_info": original_cache.get("placeholder_info"),
                    "token_ids": original_cache.get("token_ids"),
                    "input": {},
                    "response": {},
                    "response_ids": {},
                    "condition": {},
                    "condition_ids": {},
                    "input_drop_num": {},
                    "response_drop_num": {},
                    "condition_drop_num": {},
                }
                LLMChat.weight_dict = {}
            result.set_id(node_id, role)
        result.__setstate__(copied_state)
        return result
