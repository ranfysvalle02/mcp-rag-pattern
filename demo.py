"""
demo.py - Push vs. RAG vs. Pull, measured for real, fully local.

One self-contained demo, no flags. Just run it:

    ollama serve        # if it isn't already running
    python demo.py

It answers questions over a REAL corpus (the actual MCP / FastMCP docs in
./corpus) three different ways, prints the token + latency cost of each
straight from Ollama, and ends with a plain-English DIGEST of who wins when.

The three strategies:
  1. naive-push : stuff as much of the corpus into the prompt as the context
                  budget allows, answer in one call (the "just dump it" baseline).
  2. rag-push   : vector-search the top-k chunks, inject only those, one call.
                  (This is traditional RAG -- the "Push" pattern done right.)
  3. pull       : expose retrieval as an MCP tool; the model decides when and
                  what to fetch, across a multi-turn agent loop.

Pieces, all on your machine:
  - Ollama  -> local inference (qwen3:14b) + embeddings (nomic-embed-text)
  - FastMCP -> the MCP server that exposes `search_docs` and `get_live_status`
  - numpy   -> cosine similarity over the embedded corpus

Env overrides (optional):
  CHAT_MODEL=llama3.1:latest  EMBED_MODEL=nomic-embed-text:latest
  TOP_K=4  NAIVE_BUDGET=8000  (token budget the naive-push baseline may inject)
"""

from __future__ import annotations

import asyncio
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import ollama
from fastmcp import Client, FastMCP

CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3:14b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text:latest")
TOP_K = int(os.environ.get("TOP_K", "4"))

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
CACHE_PATH = os.path.join(CORPUS_DIR, ".embcache.npz")
CHUNK_CHARS = 1200  # ~300 tokens per chunk
SMALL_CTX = 8192  # context window for rag/pull (only a few chunks in play)
# Token budget the naive-push baseline is allowed to inject. Real corpora dwarf
# any context window, so "dump everything" really means "dump as much as fits".
# This also keeps the demo fast: prefilling the full corpus would take minutes.
NAIVE_BUDGET = int(os.environ.get("NAIVE_BUDGET", "8000"))

DEFAULT_QUESTION = "What transports does the MCP Python SDK support, and how do I run a server over stdio?"
# Volatile data that did not exist when any prompt was built. Push/RAG can only
# inject static docs, so they cannot answer this; only Pull can fetch it live.
LIVE_QUESTION = (
    "What is the current server time in UTC right now, and what is the system's "
    "1-minute load average?"
)

SYSTEM_PROMPT = (
    "You are a documentation assistant for the Model Context Protocol (MCP) and "
    "its Python SDK / FastMCP. Answer using ONLY the provided or retrieved "
    "documentation. Be concise, cite the specific APIs, and include short code "
    "snippets when relevant. If the docs don't cover it, say so."
)


# ---------------------------------------------------------------------------
# 1. Load the real corpus and chunk it.
# ---------------------------------------------------------------------------
def chunk_markdown(text: str, source: str, target: int = CHUNK_CHARS) -> list[dict]:
    """Greedily pack paragraphs into ~target-sized chunks, tracking the nearest
    Markdown heading so each chunk carries a little structural context."""
    chunks: list[dict] = []
    section = ""
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        body = "\n\n".join(buf).strip()
        if body:
            chunks.append({"source": source, "section": section, "text": body})
        buf, buf_len = [], 0

    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if not p:
            continue
        first = p.splitlines()[0]
        if buf_len + len(p) > target and buf:
            flush()
        buf.append(p)
        buf_len += len(p)
        if first.lstrip().startswith("#"):
            section = first.lstrip("#").strip()
    flush()
    return chunks


def load_corpus() -> list[dict]:
    paths = sorted(
        glob.glob(os.path.join(CORPUS_DIR, "*.md"))
        + glob.glob(os.path.join(CORPUS_DIR, "*.txt"))
    )
    if not paths:
        sys.exit(f"No corpus files found in {CORPUS_DIR}/ (add some .md/.txt files).")
    chunks: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            chunks.extend(chunk_markdown(f.read(), os.path.basename(path)))
    return chunks


def chunk_display(c: dict) -> str:
    """How a chunk appears once it's in the model's context."""
    head = f"{c['source']}" + (f" › {c['section']}" if c["section"] else "")
    return f"**{head}**\n\n{c['text']}"


# ---------------------------------------------------------------------------
# 2. Embed the corpus (with a real nomic task prefix), cached to disk.
# ---------------------------------------------------------------------------
def _embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        v = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"])
        n = np.linalg.norm(v)
        vecs.append(v / n if n else v)
    return np.vstack(vecs)


def embed_query(q: str) -> np.ndarray:
    v = np.array(
        ollama.embeddings(model=EMBED_MODEL, prompt=f"search_query: {q}")["embedding"]
    )
    n = np.linalg.norm(v)
    return v / n if n else v


def build_index(chunks: list[dict]) -> np.ndarray:
    """Embed all chunks, caching vectors so reruns are instant unless the corpus
    or embedding model changes."""
    payloads = [f"search_document: {c['source']} {c['section']}\n{c['text']}" for c in chunks]
    key = hashlib.sha256(("\u0000".join(payloads) + EMBED_MODEL).encode()).hexdigest()

    if os.path.exists(CACHE_PATH):
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if str(cached["key"]) == key:
            return cached["vectors"]

    print(f"[setup] embedding {len(chunks)} chunks with {EMBED_MODEL} ...")
    vectors = _embed(payloads)
    np.savez(CACHE_PATH, key=key, vectors=vectors)
    return vectors


def search(query: str, k: int = TOP_K) -> list[dict]:
    q = embed_query(query)
    scores = _VECTORS @ q
    top = np.argsort(scores)[::-1][:k]
    return [{**_CHUNKS[i], "score": float(scores[i])} for i in top]


print("[setup] loading corpus ...")
_CHUNKS = load_corpus()
_VECTORS = build_index(_CHUNKS)
_CORPUS_CHARS = sum(len(c["text"]) for c in _CHUNKS)
_SOURCES = sorted({c["source"] for c in _CHUNKS})
print(
    f"[setup] {len(_SOURCES)} docs, {len(_CHUNKS)} chunks, "
    f"~{_CORPUS_CHARS:,} chars (~{_CORPUS_CHARS // 4:,} tokens). ready.\n"
)


# ---------------------------------------------------------------------------
# 3. The MCP server. One tool: semantic search over the real corpus.
# ---------------------------------------------------------------------------
mcp = FastMCP("DocsRetrievalEngine")


@mcp.tool()
def search_docs(query: str, limit: int = TOP_K) -> str:
    """Search the MCP / FastMCP documentation and return the most relevant
    passages as Markdown. Call this whenever you need API details, transports,
    code examples, or anything about MCP or the Python SDK."""
    hits = search(query, limit)
    out = [f"### Results for: {query!r}\n"]
    for rank, h in enumerate(hits, start=1):
        head = f"{h['source']}" + (f" › {h['section']}" if h["section"] else "")
        out.append(f"**{rank}. {head}** _(score {h['score']:.2f})_\n\n{h['text']}\n")
    return "\n".join(out)


def _live_status() -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        load1, load5, load15 = os.getloadavg()
        load = f"{load1:.2f}, {load5:.2f}, {load15:.2f}"
    except (OSError, AttributeError):
        load = "unavailable"
    return f"current_utc_time: {now}\nload_average_1m_5m_15m: {load}"


@mcp.tool()
def get_live_status() -> str:
    """Return the CURRENT server time (UTC) and system load average. This data is
    live and changes on every call; use it for any real-time question."""
    return _live_status()


# ---------------------------------------------------------------------------
# 4. Metrics + the model call. Counts come straight from Ollama responses.
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    label: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0  # prefill tokens billed (re-sent history counts again)
    output_tokens: int = 0
    tool_seconds: float = 0.0
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def record(self, resp) -> None:
        self.llm_calls += 1
        self.input_tokens += int(getattr(resp, "prompt_eval_count", 0) or 0)
        self.output_tokens += int(getattr(resp, "eval_count", 0) or 0)

    def line(self) -> str:
        s = (
            f"[metrics] {self.label}: {self.llm_calls} LLM call(s), "
            f"{self.tool_calls} tool call(s) | in {self.input_tokens} tok | "
            f"out {self.output_tokens} tok | tool {self.tool_seconds:.2f}s | "
            f"wall {self.wall_seconds:.2f}s"
        )
        return s + (f"  ({'; '.join(self.notes)})" if self.notes else "")


def chat(messages: list[dict], tools: list[dict] | None = None, num_ctx: int = SMALL_CTX):
    """Call Ollama and return the full response. Disable qwen3 thinking for
    speed; fall back for models without the `think` flag (e.g. llama3.1)."""
    opts = {"temperature": 0, "num_ctx": num_ctx}
    kwargs = dict(model=CHAT_MODEL, messages=messages, options=opts)
    if tools:
        kwargs["tools"] = tools
    try:
        return ollama.chat(think=False, **kwargs)
    except Exception:
        return ollama.chat(**kwargs)


# ---------------------------------------------------------------------------
# 5. The three strategies.
# ---------------------------------------------------------------------------
def naive_push(question: str) -> tuple[str, Metrics]:
    """Inject as much of the corpus as a context budget allows, answer in one
    call. The full corpus is far bigger than any sane window, so this drops most
    of it -- which is exactly why "just dump everything" doesn't scale."""
    budget_chars = NAIVE_BUDGET * 4
    picked, used = [], 0
    for c in _CHUNKS:
        if picked and used + len(c["text"]) > budget_chars:
            break
        picked.append(c)
        used += len(c["text"])

    context = "\n\n---\n\n".join(chunk_display(c) for c in picked)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"# Documentation\n\n{context}\n\n---\nQuestion: {question}"},
    ]
    ctx = max(SMALL_CTX, NAIVE_BUDGET + 1024)

    m = Metrics(label="naive-push")
    m.notes.append(f"injected {len(picked)}/{len(_CHUNKS)} chunks")
    if len(picked) < len(_CHUNKS):
        m.notes.append(f"corpus too big for the budget -- dropped {len(_CHUNKS) - len(picked)} chunks")
    t0 = time.perf_counter()
    resp = chat(messages, num_ctx=ctx)
    m.wall_seconds = time.perf_counter() - t0
    m.record(resp)
    answer = resp["message"].get("content", "").strip()
    return answer, m


def rag_push(question: str, k: int = TOP_K) -> tuple[str, Metrics]:
    """Retrieve top-k, inject only those, answer in a single call (classic RAG)."""
    hits = search(question, k)
    context = "\n\n---\n\n".join(chunk_display(h) for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"# Retrieved context\n\n{context}\n\n---\nQuestion: {question}"},
    ]
    m = Metrics(label="rag-push")
    m.notes.append(f"retrieved {len(hits)} chunks (k={k})")
    t0 = time.perf_counter()
    resp = chat(messages)
    m.wall_seconds = time.perf_counter() - t0
    m.record(resp)
    answer = resp["message"].get("content", "").strip()
    return answer, m


def mcp_tools_to_ollama(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
        }
        for t in tools
    ]


def extract_text(result) -> str:
    blocks = getattr(result, "content", None) or []
    parts = [getattr(b, "text", "") for b in blocks if getattr(b, "text", "")]
    return "\n".join(parts) if parts else str(getattr(result, "data", result))


def parse_toolish_content(content: str, tool_names: list[str]) -> list[dict] | None:
    """llama3.1 sometimes emits a tool call as JSON in message content instead of
    the native tool_calls field. Recover those so the loop stays reliable."""
    if not content:
        return None
    s, e = content.find("{"), content.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(content[s : e + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    args = obj.get("parameters") or obj.get("arguments") or {}
    if not isinstance(args, dict) or ("query" not in args and "query" not in obj):
        return None
    if "query" in obj and not args:
        args = {k: v for k, v in obj.items() if k != "name"}
    name = obj.get("name")
    return [{"function": {"name": name if name in tool_names else tool_names[0], "arguments": args}}]


async def pull(question: str, verbose: bool = True) -> tuple[str, Metrics]:
    """Pull pattern: the model decides when/what to fetch via the MCP tool."""
    m = Metrics(label="pull")
    answer = ""
    t0 = time.perf_counter()

    async with Client(mcp) as client:
        mcp_tools = await client.list_tools()
        tools = mcp_tools_to_ollama(mcp_tools)
        tool_names = [t.name for t in mcp_tools]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if verbose:
            print(f"[agent] {CHAT_MODEL} | tools: {tool_names}")
            print(f"[user] {question}\n")

        for turn in range(1, 7):  # cap the agent loop
            resp = chat(messages, tools)
            m.record(resp)
            msg = resp["message"]
            messages.append(msg)

            calls = msg.get("tool_calls") or []
            if not calls:
                calls = parse_toolish_content(msg.get("content", ""), tool_names)
            if not calls:
                answer = msg.get("content", "").strip()
                if verbose:
                    print(f"[assistant]\n{answer}\n")
                break

            for call in calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                m.tool_calls += 1
                if verbose:
                    print(f"[pull] turn {turn}: {name}({args})")
                tt = time.perf_counter()
                result = await client.call_tool(name, args)
                m.tool_seconds += time.perf_counter() - tt
                text = extract_text(result)
                if verbose:
                    print(f"[mcp] returned {len(text)} chars\n")
                messages.append({"role": "tool", "name": name, "content": text})
        else:
            if verbose:
                print("[assistant] (stopped: hit max tool-calling turns)")

    m.wall_seconds = time.perf_counter() - t0
    return answer, m


# ---------------------------------------------------------------------------
# 6. Comparison harness.
# ---------------------------------------------------------------------------
def snippet(text: str, n: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


WIDTH = 66


def rule(title: str = "", ch: str = "=") -> None:
    if title:
        print("\n" + f" {title} ".center(WIDTH, ch))
    else:
        print(ch * WIDTH)


def print_intro() -> None:
    rule("MCP-RAG-PATTERN  -  Push vs RAG vs Pull, measured live")
    print("Three ways to get knowledge into an LLM, answered over the REAL")
    print("MCP / FastMCP docs, with the real token + latency cost of each:\n")
    print("  naive-push  stuff as much of the corpus into the prompt as fits")
    print("  rag-push    retrieve the top-k relevant chunks, inject only those")
    print("  pull        give the model an MCP search tool; it fetches on demand\n")
    print("Legend:  [pull] model chose to call a tool   [mcp] tool result")
    print("         [metrics] real tokens + wall-clock, straight from Ollama")
    rule()


def verdict(label: str, metrics: list[Metrics]) -> str:
    cheapest = min(metrics, key=lambda m: m.input_tokens).label
    fastest = min(metrics, key=lambda m: m.wall_seconds).label
    tags = []
    if label == cheapest:
        tags.append("fewest tokens")
    if label == fastest:
        tags.append("fastest")
    return " + ".join(tags)


def print_digest(question: str, static: list[Metrics]) -> None:
    naive, rag, pull_m = static
    cheapest = min(static, key=lambda m: m.input_tokens)
    fastest = min(static, key=lambda m: m.wall_seconds)

    rule("DIGEST  -  what just happened", "=")
    print(f"Corpus : {len(_SOURCES)} docs, {len(_CHUNKS)} chunks, "
          f"~{_CORPUS_CHARS // 4:,} tokens (real MCP/FastMCP docs)")
    print(f"Model  : {CHAT_MODEL}  (local, via Ollama)\n")

    print(f'Static-doc question: "{snippet(question, 60)}"')
    print(f"  {'strategy':<12}{'in tok':>8}{'out tok':>8}{'calls':>8}{'wall':>8}   note")
    print("  " + "-" * (WIDTH - 2))
    for m, note in [
        (naive, f"saw {naive.notes[0].split()[1]} of corpus; the wasteful baseline"),
        (rag, "retrieve-then-inject, one shot"),
        (pull_m, f"{pull_m.tool_calls} tool call(s), agent-driven"),
    ]:
        calls = f"{m.llm_calls}L/{m.tool_calls}T"
        flag = verdict(m.label, static)
        tail = note + (f"  <- {flag}" if flag else "")
        print(f"  {m.label:<12}{m.input_tokens:>8,}{m.output_tokens:>8,}"
              f"{calls:>8}{m.wall_seconds:>7.1f}s   {tail}")
    print(f"\n  => cheapest: {cheapest.label} ({cheapest.input_tokens:,} tok)   "
          f"fastest: {fastest.label} ({fastest.wall_seconds:.1f}s)")

    rule("TAKEAWAYS", "-")
    print("  - Don't dump the whole corpus: naive-push costs the most for a partial view.")
    print("  - Static knowledge     -> RAG (retrieve-then-inject): one call, fewest tokens.")
    print("  - Live / stateful data -> Pull (MCP tools): often the ONLY correct option.")
    print("  - Hybrid = RAG wrapped in an MCP tool: RAG's efficiency + the model's agency.")
    rule()


async def run_demo() -> None:
    print_intro()

    rule("SCENARIO 1  -  a static-doc question", "-")
    print(f'Q: "{DEFAULT_QUESTION}"\n')

    print("[1/3] naive-push  (inject as much of the corpus as the budget allows)")
    a_naive, m_naive = naive_push(DEFAULT_QUESTION)
    print(f"  -> {snippet(a_naive)}")
    print("  " + m_naive.line() + "\n")

    print("[2/3] rag-push  (retrieve top-k, inject only those)")
    a_rag, m_rag = rag_push(DEFAULT_QUESTION)
    print(f"  -> {snippet(a_rag)}")
    print("  " + m_rag.line() + "\n")

    print("[3/3] pull  (the model calls an MCP search tool on its own)")
    _, m_pull = await pull(DEFAULT_QUESTION, verbose=True)
    print("  " + m_pull.line())

    print_digest(DEFAULT_QUESTION, [m_naive, m_rag, m_pull])


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
