import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { createInterface } from "node:readline";

export async function loadJsonl(filePath) {
  const rows = [];
  const stream = createReadStream(filePath, { encoding: "utf8" });
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    rows.push(JSON.parse(trimmed));
  }
  return rows;
}

export async function loadJson(filePath) {
  const raw = await readFile(filePath, "utf8");
  return JSON.parse(raw);
}

export async function appendJsonl(filePath, record) {
  const { appendFile } = await import("node:fs/promises");
  await appendFile(filePath, `${JSON.stringify(record)}\n`, "utf8");
}
