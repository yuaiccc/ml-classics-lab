// 浏览器端实时调用本地 Ollama（经 Vite 同源代理 /ollama → localhost:11434，避免跨域）。
// 大模型实验靠它真·实时跑，而不是回放预计算 JSON。
const BASE = "/ollama";

export async function embed(model: string, input: string[]): Promise<number[][]> {
  const r = await fetch(`${BASE}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input }),
  });
  if (!r.ok) throw new Error(`Ollama embed 失败 (${r.status})`);
  const d = await r.json();
  return d.embeddings as number[][];
}

export async function generate(
  model: string,
  prompt: string,
  options?: { temperature?: number; num_predict?: number; stop?: string[] }
): Promise<string> {
  const r = await fetch(`${BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, prompt, stream: false, options }),
  });
  if (!r.ok) throw new Error(`Ollama generate 失败 (${r.status})`);
  const d = await r.json();
  return ((d.response as string) || "").trim();
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

/** 检查本地 Ollama 是否可达（经代理） */
export async function ollamaReachable(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/api/tags`, { signal: AbortSignal.timeout(4000) });
    return r.ok;
  } catch {
    return false;
  }
}
