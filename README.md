# mcp-rag-pattern

A reference for **choosing between retrieval strategies** for LLMs — Context
Injection (Push), traditional RAG, and tool-based retrieval via the Model
Context Protocol (Pull) — backed by a fully local demo that **measures the real
trade-offs** instead of asserting them.

- **[`blog.md`](./blog.md)** — the deep dive: Push vs. Pull, the MCP primitives
  and transports, a reference matrix, and a measured cost/speed analysis.
- **[`demo.py`](./demo.py)** — a runnable benchmark. It builds a corpus from the
  **real MCP / FastMCP documentation**, embeds it locally, and answers questions
  under three strategies, reporting the token counts and latency Ollama actually
  returns.
- **[`corpus/`](./corpus)** — the real source docs (MCP Python SDK, servers, and
  FastMCP READMEs). Drop in more `.md`/`.txt` files to grow it.

## The three strategies

| Strategy | How it retrieves | LLM calls |
| :--- | :--- | :--- |
| `naive-push` | stuff as much of the corpus into the prompt as fits | 1 |
| `rag-push` | vector-search top-_k_, inject only those (classic RAG) | 1 |
| `pull` | expose search as an **MCP tool**; the model fetches on demand | 2+ |

## Quickstart

Requires [Ollama](https://ollama.com) running locally.

```bash
# 1. Pull the models the demo uses
ollama pull qwen3:14b         # chat / tool-calling agent (default)
ollama pull nomic-embed-text  # embeddings for vector search

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run it -- no flags, one command, ends with a digest
python demo.py
```

One command runs the whole story: a static-doc question through all three
strategies, then a plain-English **digest** of who wins when. Takes ~70s on a
laptop. The corpus is embedded once and cached to
`corpus/.embcache.npz`, so reruns are fast. Everything runs on your machine: no
API keys, no network calls, no ports — the agent connects to the FastMCP server
in-process.

### Optional env overrides

```bash
CHAT_MODEL=llama3.1:latest python demo.py   # llama works too (JSON tool-call fallback handled)
EMBED_MODEL=nomic-embed-text python demo.py
TOP_K=6 python demo.py                       # chunks retrieved by rag/pull
NAIVE_BUDGET=8000 python demo.py             # token budget the naive baseline may inject
```

## What the numbers say (measured, `qwen3:14b`, ~25.8k-token corpus)

The run ends with a digest like this:

```
================= DIGEST  -  what just happened ==================
Corpus : 3 docs, 96 chunks, ~25,758 tokens (real MCP/FastMCP docs)
Model  : qwen3:14b  (local, via Ollama)

Static-doc question: "What transports does the MCP Python SDK support, ..."
  strategy      in tok out tok   calls    wall   note
  ----------------------------------------------------------------
  naive-push     8,060     167   1L/0T   33.5s   saw 28/96 of corpus; the wasteful baseline
  rag-push       1,145     245   1L/0T   13.2s   retrieve-then-inject  <- fewest tokens + fastest
  pull           3,440     347   2L/2T   20.9s   2 tool call(s), agent-driven

  => cheapest: rag-push (1,145 tok)   fastest: rag-push (12.5s)
```

- **Naive Push is the anti-pattern**: most tokens, slowest — and even at an 8k
  budget it can only see ~30% of the corpus. "Dump everything" doesn't scale.
- **For static-doc Q&A, RAG wins**: one retrieval, one call, cheapest, fastest.
- **Pull costs a bit more for the same answer** (tool-schema tax + a round trip) —
  you're paying for agency you didn't need on a simple lookup.

The real lesson: **RAG for static knowledge; Pull (MCP) when the data is live,
stateful, mutating, or unpredictable** — see [`blog.md` §4–5](./blog.md) for the
volatile-data case, plus what happens at scale with hundreds of tools and a
document-native gateway.
