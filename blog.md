# Context Injection vs. Model Context Protocol (MCP): The Architecture Battle for LLM Retrieval

> **Engineering & Architecture Deep Dive · June 2026**

As the context windows of Large Language Models (LLMs) expanded from 4k tokens to well over 1 million, a naive architectural thesis emerged: *"Just dump everything into the prompt and let the model sort it out."* But production systems operating at scale quickly collided with the harsh realities of token costs, "Lost in the Middle" retrieval degradation, and latency constraints.

This reality has triggered an intense debate among AI engineers about data-retrieval design patterns. Specifically: should we proactively inject data as Markdown straight into the initial context window (the **"Push"** pattern), or expose data infrastructure as dynamic tools via the Model Context Protocol (the **"Pull"** pattern)?

The short answer is that neither is an anti-pattern. They represent distinct architectural choices with fundamental trade-offs. This post walks the full arc: the mechanics of each pattern, a **measured** head-to-head benchmark (real token counts and latencies, run locally), the hybrid design that production teams have converged on — and finally, what happens to all of this *at scale*, when an agent juggles hundreds of tools and the question stops being "which retrieval pattern?" and becomes "what is the right storage substrate for a protocol made of documents?"

---

## 1. The "Push" Pattern: Upfront Markdown Context Injection

The Push pattern is the standard design of traditional Retrieval-Augmented Generation (RAG). In this paradigm, the orchestration layer intercepts the user's query, conducts a programmatic lookup (a semantic vector search, a full-text query, or a metadata filter), formats the returned data chunks, and stitches them directly into the system or user prompt **before** the LLM executes its first token.

### Why Markdown Is the Native Language of Context

When injecting context upfront, structuring the data as clean Markdown has become the industry standard. Modern LLMs are trained extensively on code repositories and web text where structural hierarchy is denoted with Markdown syntax. LLMs excel at parsing:

- **Heading levels** (`#`, `##`, `###`) to distinguish document boundaries and subsections.
- **Data tables** to isolate structured relational rows without losing semantic relations.
- **Fenced code blocks** (e.g. ```` ```typescript ````) to prevent variable confusion in code-comprehension pipelines.

> 💡 **Architectural Advantage: Single-Turn Determinism**
> Because the orchestration layer injects all data upfront, the LLM executes in a single turn. There are no intermediate tool calls, zero agent loops, and minimal round-trip latency over the network.

### When It Becomes an Anti-Pattern

Upfront context injection degrades rapidly when used indiscriminately. Shoveling massive blocks of unstructured text into a prompt "just in case" introduces three problems:

1. **Attention Attenuation ("Lost in the Middle").** Despite high context limits, models still exhibit diminished attention density toward information placed in the *middle* of a long prompt. This was first formalized by Liu et al. (2023) and remains a measurable effect even in frontier long-context models.
2. **Linear Cost Escalation.** Processing thousands of unused tokens on *every* prompt compounds costs, particularly with top-tier reasoning models.
3. **Stale State.** Once data is injected into the context window, it is frozen. In a multi-turn troubleshooting session, the data injected in turn one may be inaccurate by turn five.

---

## 2. The "Pull" Pattern: Tool-Based Retrieval via Model Context Protocol (MCP)

The Pull pattern shifts control from the application orchestration layer to the LLM itself. Instead of guessing what data the model needs and forcing it into the prompt upfront, the application exposes data repositories as executable tools or resources, letting the model request data on an as-needed basis.

Function calling has existed for years, but the architecture underwent a dramatic standardization with the release and wide adoption of the **Model Context Protocol (MCP)**, originally open-sourced by Anthropic. Inspired heavily by the developer ecosystem's Language Server Protocol (LSP), MCP establishes an open standard that creates an abstraction layer between LLM **hosts** (like Claude Code, ChatGPT Desktop, or custom enterprise applications), their **clients** (the connectors inside a host), and external **servers** (the data and tool infrastructure).

### The Core Primitives of MCP

The MCP specification defines three server-side primitives, distinguished by *who* controls invocation:

- **Tools (model-controlled).** Dynamic, executable functions exposed to the LLM. The model decides *if*, *when*, and *with what arguments* to execute them. Tools can perform calculations, read state, and execute read-write mutations (e.g. `update_ticket_status`).
- **Resources (application-controlled).** Standardized URI-addressable read-only endpoints (e.g. `logs://production/errors`), analogous to a `GET` in REST. The host application or user explicitly loads resources into the conversation stream, keeping control over context boundaries.
- **Prompts (user-controlled).** Reusable templated messages and workflows that a user selects to invoke tools and resources in an optimal, pre-defined way.

> The original article framed MCP as a strict "Tools vs. Resources" dichotomy. In practice the spec defines **three** primitives — Tools, Resources, and Prompts — plus client-offered capabilities such as Sampling, Roots, and Elicitation.

### Why the Industry Converged on MCP

Before MCP, connecting *N* custom backends or internal tools to *M* different LLM providers required writing *N × M* unique integration wrappers. MCP collapses this topology into a standard client-server architecture, with all messages encoded as **JSON-RPC 2.0**. The spec defines two standard transport bindings:

- **stdio** — newline-delimited JSON-RPC over the standard streams of a client-launched subprocess. Ideal for local integrations.
- **Streamable HTTP** — each message is an HTTP `POST` to a single MCP endpoint; replies arrive either as a JSON object or as a request-scoped Server-Sent Events (SSE) stream.

> ⚠️ **Spec note (current as of 2026):** The earlier standalone **HTTP+SSE** transport has been **superseded by Streamable HTTP** (introduced in the 2025-03 spec) and is now classified as *deprecated*. SSE is no longer a transport in its own right — it survives only as one of the two reply modes inside Streamable HTTP. If you read older posts describing "HTTP + SSE" as the remote transport, treat that as legacy terminology.

> ⚠️ **Architectural Cost: The Latency Tax**
> The Pull pattern is highly agentic — it forces a multi-turn loop. The model must halt generation, emit a tool-execution payload, wait for the MCP server to run the underlying lookup, process the returned string, and run a second generation cycle. This can increase perceived user latency by **2×–4×**.

---

## 3. Architectural Reference Matrix

| Dimension | Upfront Markdown Injection (Push) | Tool / MCP-Driven Retrieval (Pull) |
| :--- | :--- | :--- |
| **Execution Timing** | Before the LLM call begins. Orchestration manages routing. | During the generation lifecycle. The model manages routing. |
| **Latency Profile** | Optimized (low): single LLM API transaction. | Variable (high): multi-turn loops and execution pauses. |
| **Data Modality** | Best for massive, static, unstructured corpuses (wikis, PDFs, docs). | Best for live, highly mutable, structured data (logs, production DBs, Slack APIs). |
| **State Capabilities** | Read-only: injected blocks are static for the turn. | Read/write: pull data, mutate state, commit changes back. |
| **Token Efficiency** | Poor if chunks are broad/blind. High waste risk. | High: the model pulls specific slices strictly when needed. |

---

## 4. Balancing the Budget: Speed, Token Bloat, and Cost

The matrix above lists "latency" and "token efficiency" as single cells, but in practice these are the two dials you will actually argue about in design review. They pull against each other, and the right call depends on *how often* a given query needs the data.

### Speed: One Round Trip vs. Many

Wall-clock latency for an LLM call breaks down into three buckets: **time-to-first-token (TTFT)**, which is dominated by *prefill* (the model reading every input token); **decode** (generating output tokens one at a time); and, for the Pull pattern, **tool execution + network round trips**.

- **Push** pays one prefill, then decodes once — a single API transaction, no pauses. But "Push" splits in two: *naive* Push ingests the **entire** corpus (a giant prefill), while **RAG** — Push done right — retrieves the top-_k_ chunks first and injects only those (a small prefill). The difference is enormous, and §4 measures it: 24.5k vs. 1.1k input tokens, 123s vs. 13s, on the *same* model and question.
- **Pull** pays a *small* prefill per call but multiplies the number of calls. A single tool use is a minimum of **two** generation cycles (decide-to-call → read-result → answer), plus the server's own lookup time. That is the "latency tax" from §2. For a single *local* tool call it is mild (we measured ~1.3× over RAG); it climbs toward 2×–4× when retrieval is remote or the model makes several hops.

The non-obvious part: prefill compute scales super-linearly with input length, so *naive* Push is the latency disaster, not Pull — dumping the whole corpus "just in case" balloons TTFT far past what a small retrieval plus a round trip would cost. The latency winner is whichever pattern keeps the prefill *small and relevant*, which is exactly what both RAG and Pull do.

### Token Bloat: What You Pay For, and How Often

This is where the two patterns diverge hardest, especially in multi-turn sessions:

- **Push re-bills the blob.** Whatever you stitch into the prompt is part of the input on **every turn it stays in history**. Inject 8k tokens of docs on turn one of a ten-turn chat and you can pay for them ten times unless you actively prune. Reasoning models make this worse because thinking tokens are billed too.
- **Pull pays a fixed schema tax plus only what it fetches.** Every tool definition you expose is serialized into the request on *every* call — a real, recurring cost that grows with the number of tools. In exchange, the model only pulls the specific slices it needs, when it needs them. In the demo, Pull carried ~1,991 input tokens for a single-hop answer versus RAG's ~1,145 — the extra ~850 tokens are precisely that schema tax plus the re-sent first turn (see §4).

**Prompt caching changes the math.** Most providers now cache a static prompt prefix and bill cache hits at a steep discount (often ~10% of the input rate). That heavily favors **Push when the injected context is stable across turns** — the expensive blob is paid once at full price, then cheaply thereafter. Volatile data breaks the cache on every change, which tilts back toward **Pull**.

### Measured: Three Real Strategies on a Real Corpus

The numbers below are not estimates. To replace assertion with measurement, I built a small, fully-local benchmark: a corpus assembled from the **actual MCP / FastMCP documentation** (the real Python SDK, servers, and FastMCP READMEs — 3 docs, 96 chunks, ~25.8k tokens), embedded with a local `nomic-embed-text`, and answered by `qwen3:14b` (temperature 0) under three strategies. The token counts and timings below come straight from the model runtime. (A complete, runnable reference implementation accompanies this article; everything here reproduces on a laptop.)

Crucially, this distinguishes the *naive* Push strawman (dump everything) from **RAG** — the Push pattern done right (retrieve top-_k_, then inject in a single call):

**Single-hop question** — *"What transports does the MCP Python SDK support, and how do I run a server over stdio?"*

| Metric | naive-push (8k budget) | rag-push (top-k + inject) | pull (MCP tool) |
| :--- | ---: | ---: | ---: |
| LLM calls | 1 | 1 | 2 |
| Tool calls | 0 | 0 | 2 |
| Input tokens | **8,060** | **1,145** | **3,440** |
| Output tokens | 167 | 245 | 347 |
| Wall-clock | **33.5s** | **13.2s** | **20.9s** |
| Corpus seen | 28 / 96 chunks | 4 chunks | model's choice |

Three findings fall straight out of the measurements:

1. **Naive Push doesn't scale.** A real corpus dwarfs any context window, so "dump everything" really means "dump as much as fits." Even capped at an 8k-token budget, naive-push is the most expensive and slowest path here — *and it still sees only ~30% of the corpus.* Remove the cap and force the entire 25.8k-token corpus into a 40k window and it is far worse: **24,538 input tokens and ~123 seconds** — 21× the tokens and ~10× the latency of RAG, for a worse-organized answer. "Just dump it in the context window" is the thing to avoid.
2. **For static-document Q&A, RAG (Push done right) wins.** One retrieval, one LLM call, the fewest tokens, the lowest latency. If you know you'll need the docs, fetching them *before* generation beats making the model ask.
3. **Pull costs more for the same answer.** It used ~3× the input tokens of RAG (3,440 vs. 1,145) and ~1.6× the latency (20.9s vs. 13.2s), because it pays a fixed tool-schema tax, re-sends history across turns, and — on this pass — chose to run *two* searches. That gap is the price of *agency*, and on a single-hop fact lookup you're paying for agency you didn't need.

There is also a subtler cost: **Pull's price is non-deterministic.** The same transports question sometimes yields one search (~1,991 tokens) and sometimes two (3,440 tokens) — because the model, not your code, decides how many times to fetch. Multi-part wording neither guarantees nor forbids multiple retrievals; the model chooses. That flexibility is the feature; the variable, harder-to-budget cost is the bill. RAG's cost, by contrast, is fixed by your top-_k_.

> The latency story also corrects the §2 headline. Pull's tax over RAG was a mild **~1.3×** for a single in-process tool call — nowhere near 2×–4×, which assumes remote, multi-hop retrieval. The catastrophic latency here belonged to *naive Push* (10×), not to tools.

### Where Pull Actually Wins: Volatile and Stateful Data

If RAG beats Pull on static-doc Q&A, why expose tools at all? Because some questions **cannot be answered from anything you injected**, no matter how much you inject. A second experiment asks for live data — the current server time and system load:

```text
# LIVE QUESTION: What is the current server time in UTC right now, and the 1-minute load average?
# (ground truth right now: current_utc_time: 2026-06-25T06:14:34Z | load_average: 1.95, 2.30, 2.70)

--- rag-push (static docs only) ---
[answer] To retrieve the current server time ... you can use the `time` and `system` tools ...
         ```python
         @mcp.tool()
         def get_server_time_and_load(ctx: Context): ...
         ```
[metrics] rag-push: 1 LLM call | in 899 tok | out 201 tok | wall 7.9s

--- pull (can fetch live data via tool) ---
[pull] turn 1: get_live_status({})
[mcp] returned 91 chars
[assistant] The current server time in UTC is 2026-06-25T06:14:44Z, and the 1-minute load average is 2.03.
[metrics] pull: 2 LLM calls, 1 tool call | in 730 tok | out 76 tok | wall 3.1s
```

This is a **categorical** difference, not a cost one. RAG never had the live value in its context, so it did the only thing it could: it hallucinated *instructions for how you might fetch it* and never produced an answer. Pull called the tool and returned the correct time and load (matching ground truth within seconds). When the data is volatile, stateful, or behind a live system, the token table is irrelevant — only Pull can be *correct*. The same logic extends to write actions: RAG can describe how to reset an API key; only a tool can actually do it.

### A Decision Heuristic, Grounded in the Numbers

The measurements collapse into a short decision tree. The first axis is **whether the data is static or live**; the second is **whether you can predict what's needed before generation**:

- **Never naive-push at scale.** Injecting the whole corpus was 21× the tokens and 10× the latency of RAG for a *worse* answer. The only time "inject everything" is acceptable is when "everything" is genuinely small (a few hundred tokens) and stable.
- **Static data + predictable need → RAG (Push done right).** If you can guess what the query needs, retrieve the top-_k_ and inject it in one call. It was the cheapest *and* fastest option in every static-doc test. Prompt caching tilts this even further when the injected prefix repeats across turns.
- **Live / stateful data, or write actions → Pull (tools / MCP).** When the answer didn't exist at build time — current time, live metrics, a row that just changed, an action that must be executed — Pull isn't cheaper or pricier, it's the *only correct* option. RAG can only describe how it *would* fetch the data.
- **Unpredictable or unbounded retrieval → Pull (or hybrid).** When you genuinely cannot pre-plan which slices are needed, the corpus dwarfs the context window, or the query may require several distinct lookups, hand routing to the model. You pay the schema tax and round trips, but you avoid injecting a giant mostly-unused blob — and you scale past what fits in context.

> The trap to avoid: reaching for an MCP tool because it feels "agentic," when a plain RAG retrieval would have answered the same static question in one cheaper, faster call. Agency is a cost; only pay it when the workload (freshness, state, mutation, unpredictability) actually demands it.

### Practical Levers

Whichever side you land on, these knobs move the cost/latency curve the most:

| Lever | Effect |
| :--- | :--- |
| **Prompt caching** | Slashes the re-billing cost of a stable injected prefix (favors Push). |
| **Chunk size + top-*k*** | Smaller, fewer, higher-precision chunks cut both prefill cost and "Lost in the Middle" risk. |
| **Return compressed Markdown, not raw rows** | A tool that summarizes server-side hands back fewer tokens than dumping a SQL result set. |
| **Cap the tool count** | Fewer tool schemas means a smaller fixed tax on every request. |
| **Bound the agent loop** | A hard turn limit (the demo caps at 6) prevents a runaway model from racking up round trips. |

> **Rule of thumb:** optimize for *tokens-in-context-per-useful-answer*, not for total tokens available. Push wins when the useful fraction is high and stable; Pull wins when it is low, volatile, or unboundedly large.

---

## 5. The Modern Meta: The Hybrid Retrieval Design Pattern

In mature production architectures built throughout late 2025 and 2026, top-tier engineering teams rarely choose between these patterns in isolation. Instead they implement a **Hybrid Retrieval Strategy** that wraps traditional vector-search pipelines inside a standardized MCP server layer.

This lets an autonomous agent evaluate a complex query, determine exactly which sub-slices of documentation it needs, invoke an MCP tool like `search_docs(query: str)`, have the server run the dense vector semantic search, compress the results into clean semantic Markdown, and hand them back into the model context dynamically.

Consider this minimal Python MCP retrieval server using the FastMCP framework:

```python
from mcp.server.fastmcp import FastMCP
import database_provider as db

# Initialize standard MCP server container
mcp = FastMCP("ProductionRetrievalEngine")


@mcp.tool()
def retrieve_customer_context(customer_id: str, dynamic_range: str = "30d") -> str:
    """
    Queries the central transactional ledger and vector store for customer activity.
    Returns clean, hierarchically formatted Markdown text.
    """
    # 1. Fetch live metrics (the dynamic component)
    metrics = db.get_live_metrics(customer_id, dynamic_range)

    # 2. Compile a structured Markdown payload
    markdown_output = f"### Customer Context for ID: {customer_id}\n"
    markdown_output += "| Metric | Value |\n| :--- | :--- |\n"
    markdown_output += f"| Health Score | {metrics['score']}% |\n"
    markdown_output += f"| Avg Response Latency | {metrics['latency_ms']}ms |\n\n"

    # 3. Pull historical semantic context
    historical_chunks = db.query_vector_store(customer_id, limit=3)
    markdown_output += "#### Critical Interaction History\n"
    for chunk in historical_chunks:
        markdown_output += f"- **{chunk['date']}**: {chunk['summary']}\n"

    return markdown_output


if __name__ == "__main__":
    # Execute standalone over standard I/O (stdio) transport
    mcp.run(transport="stdio")
```

> **Packaging note:** The import above (`from mcp.server.fastmcp import FastMCP`) targets **FastMCP 1.0**, which was folded into the official MCP Python SDK in 2024 and is perfectly valid. For new projects the **actively maintained standalone `fastmcp` package** (`from fastmcp import FastMCP`) is recommended — it tracks the spec more aggressively and adds transports, auth, and a built-in test client.

### Seeing It Run, Fully Local

The reference implementation runs this end to end with no API keys and no network calls: a FastMCP server exposes a `search_docs` tool over the embedded MCP/FastMCP docs, and a local `qwen3:14b` model (served by [Ollama](https://ollama.com), with `nomic-embed-text` for embeddings) decides for itself when to pull. Here is the Pull loop in action — the agent inspects the question, calls the tool with its *own* query, then composes a grounded answer:

```text
[setup] 3 docs, 96 chunks, ~103,033 chars (~25,758 tokens). ready.

[agent] qwen3:14b | tools: ['search_docs', 'get_live_status']
[user] What transports does the MCP Python SDK support, and how do I run a server over stdio?

[pull] turn 1: search_docs({'query': 'transports supported by MCP Python SDK', 'limit': 5})
[mcp] returned 6283 chars

[pull] turn 1: search_docs({'query': 'run a server over stdio', 'limit': 5})
[mcp] returned 5775 chars

[assistant]
The MCP Python SDK supports the following transports:
1. Streamable HTTP - recommended; supports JSON or SSE response formats ...
2. SSE (Server-Sent Events) - being superseded by Streamable HTTP ...
To run a server over stdio, you can use the `stdio_server` function from the
`mcp.server.stdio` module:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, InitializationOptions(...))

[metrics] pull: 2 LLM call(s), 2 tool call(s) | in 3440 tok | out 347 tok | tool 0.14s | wall 17.83s
```

The `[pull]` lines are the whole thesis in one place: the model paused, decided *on its own* to issue **two** searches (transports, and stdio), the MCP server ran each semantic search and returned Markdown, and a second cycle composed the answer. The `[metrics]` footer is what turns this from a demo into evidence — every run reports its real token and latency cost, so the trade-offs from §4 are reproducible rather than rhetorical.

---

## 6. Scaling the Pull Pattern: The MCP Gateway

The benchmark above hides a scaling wall behind its simplicity: it exposes exactly **one** tool. Real agents don't live that way. A production host connects to many MCP servers — a GitHub server, a database server, a Slack server, an internal-wiki server — and accumulates *dozens to hundreds* of tools. That changes the economics in a way that should feel familiar by now.

### The Token Bloat Comes Back — One Layer Up

Every tool definition is JSON Schema, and every schema is serialized into the request **on every turn**. The naive move is to hand the model the entire menu each time: all two hundred definitions, every nested input schema, dumped into the context before the agent does anything useful. As the [oblivio.company essay on MCP and document storage](https://blog.oblivio-company.com/posts/6a30bb321df9feb4783617ce) puts it, that is *"twenty thousand tokens of throat-clearing per request."*

Look closely and it is **exactly the naive-Push anti-pattern from §4**, relocated from the document corpus to the tool catalog. Same disease: you pay, on every call, for context the model will almost never use. Same attention dilution. Same linear cost escalation. The fact that it recurs at the tooling layer is the tell that Push-vs-Pull is not a one-off retrieval decision — it is a general law about what you put in a context window.

### Tool Selection Is a Retrieval Problem

If the disease is the same, so is the cure: **stop treating the tool list as a list and start treating it as a search.** Embed every tool's name and description into a vector, embed the agent's intent, and return only the handful of tools that are actually relevant to *this* turn. The token bill for tool definitions collapses by an order of magnitude, and the model is no longer choosing from a wall of two hundred near-duplicates.

This is RAG — the exact same `rag-push` that won §4 — applied to the tools themselves. Sometimes called *dynamic tool discovery* or "tool RAG," it is the layer an **MCP gateway** provides: a single endpoint that sits between the host and the fleet of downstream servers, aggregates their catalogs, and hands the model a *retrieved* subset of tools instead of the whole inventory.

> Semantic search alone has a cruel blind spot, and the oblivio essay names it well: vectors are brilliant at *intent* and clumsy with *literals*. Ask for the tool literally named `getCurrentWeather`, or to look up order `A-417`, and cosine similarity may sail right past the exact token, because it rewards meaning over spelling. The fix is **hybrid search**: run a lexical (BM25-style) query *and* a vector query, then fuse the rankings. Keyword precision and semantic recall cover for each other — the difference between a router that usually works and one you can trust.

### Where It All Lives: MCP Is Document-Native

This raises a storage question that the oblivio essay answers with a genuinely elegant observation. Strip MCP to its core and a request is a JSON-RPC envelope: a `method`, an `id`, and a `params` object **whose shape depends entirely on which tool you're calling**. A weather call carries a city; an order lookup carries an id; a deploy tool carries a twelve-field config. The tool definitions themselves are nested JSON Schemas. Errors are structured objects. Try to force that into fixed rows and columns and you fight the shape forever — flattening into sparse columns, or surrendering to an opaque `JSONB` blob and migrating the schema every time a new server appears.

But the shape is already familiar: nested, self-describing, free to differ from its neighbors. **It's a document.** A tool-catalog entry is just:

```json
{
  "name": "get_current_weather",
  "server": "weather",
  "description": "Get the current weather for a city",
  "inputSchema": { "city": "string", "unit": "string" },
  "scopes": ["weather", "readonly"],
  "embedding": [0.0231, -0.0142, 0.0087, "… 768 dims"]
}
```

An embedding is just an array of floats — native JSON. The vector belongs in the *same* document as the description it describes, beside the scopes and the input schema. So a document database such as **MongoDB** stores the protocol's native unit with no impedance mismatch: the registry of servers, the tool catalog, the ephemeral session state, and the audit trail of every call are all documents, because they were all *born* as JSON the moment they crossed the wire. A new server with an unfamiliar tool shape is a `db.insert`, not an `ALTER TABLE`.

The real payoff is that hybrid tool-routing becomes **one query against one collection** — no separate vector store, no separate search engine, no sync job for the two to drift apart at 3 a.m. MongoDB's [`$rankFusion`](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/) stage (Reciprocal Rank Fusion, MongoDB 8.0+) fuses a `$vectorSearch` arm and a `$search` arm server-side:

```js
db.tool_catalog.aggregate([
  { $rankFusion: {
      input: { pipelines: {
        semantic: [
          { $vectorSearch: {
              index: "tool_vec", path: "embedding",
              queryVector: embed(intent), numCandidates: 100, limit: 20 } }
        ],
        lexical: [
          { $search: { index: "tool_text",
              text: { query: intent, path: ["name", "description"] } } },
          { $limit: 20 }
        ]
      } },
      combination: { weights: { semantic: 0.6, lexical: 0.4 } }
  } },
  { $match: { scopes: callerScope } },   // authz is a filter on the same query
  { $limit: 5 }                          // the few tools this turn actually needs
])
```

Meaning, spelling, and the metadata that says *who's allowed to call this tool* are evaluated together, because they are all just fields on the same document. (`$rankFusion` is a preview feature; the same idea is achievable today by fusing two queries in the application layer.)

### The Hybrid Pattern, All the Way Down

Step back and the symmetry is the whole point. At the **data layer**, you wrap a vector pipeline in an MCP server so the agent can pull document chunks on demand. At the **tooling layer**, you wrap a vector-plus-lexical pipeline in an MCP *gateway* so the agent can pull the right *tools* on demand. Both are RAG. Both fight the same token-bloat anti-pattern. And both are most naturally backed by a store whose unit of storage is the protocol's unit of exchange — a document. The Pull pattern doesn't just retrieve your data; at scale, it retrieves its own capabilities.

---

## 7. Conclusion & Strategic Guidance

The measured results in §4 reduce to four engineering rules of thumb:

- **Never naive-push at scale.** Injecting an entire corpus "just in case" was 21× the tokens and 10× the latency of RAG for a worse answer. Reserve full injection for genuinely tiny, stable context.
- **Default to RAG for static knowledge.** Retrieve top-_k_, inject in one call. It was the cheapest and fastest option for every static-document question, and prompt caching widens its lead when the prefix repeats.
- **Reach for MCP tools when the workload demands agency, not because it feels modern.** Live or stateful data, write actions, multi-source routing, or retrieval you cannot pre-plan are the cases where Pull is the *only* correct choice — and where the round-trip tax is worth paying. For a plain static-doc lookup it is overhead.
- **Adopt the hybrid pattern at scale.** Wrap your vector pipeline in an MCP server that returns contextualized Markdown. You get RAG's efficiency, the model's ability to decide when a pull is warranted, and standard cross-platform modularity — all reproducible end to end with the accompanying reference implementation.
- **Route tools the way you route documents.** Past a few dozen tools, stop injecting the whole catalog. Retrieve the relevant tools per turn with hybrid (semantic + lexical) search behind an MCP gateway, and store the catalog where the protocol's JSON already lives — a document database — so registry, config, routing, and audit are one backend, not four (§6).

---

## Appendix: Specifications & References

1. **Model Context Protocol — Core Specification.** Official protocol framework and documentation. <https://modelcontextprotocol.io/docs/getting-started/intro>
2. **MCP Python SDK Repository.** Official implementation schema and FastMCP quickstart guides. <https://github.com/modelcontextprotocol/python-sdk>
3. **Open-Source Reference Servers.** Curated implementations for databases, filesystems, and API integration layers. <https://github.com/modelcontextprotocol/servers>
4. **MCP Resources — Feature Architecture & Design.** Application-controlled contextual data delivery vs. model-controlled tool actions. Zuplo (2025). <https://zuplo.com/blog/mcp-resources>
5. **Threat Modeling & Security Primitives in MCP.** Prompt-injection vectors and runtime isolation in cross-system LLM tool calls. Unit 42 Research (2025). <https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/>
6. **MCP Transports Specification.** Standard transport bindings (stdio, Streamable HTTP). <https://modelcontextprotocol.io/specification/draft/basic/transports>
7. **Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023).** <https://arxiv.org/abs/2307.03172>
8. **FastMCP (standalone, actively maintained).** Docs: <https://gofastmcp.com> · Repo: <https://github.com/PrefectHQ/fastmcp>
9. **"Your AI Agent Speaks JSON. Why is Your Database Speaking Rows?"** The document-native shape of MCP and the gateway storage model. oblivio.company (2026). <https://blog.oblivio-company.com/posts/6a30bb321df9feb4783617ce>
10. **MongoDB `$rankFusion` (Reciprocal Rank Fusion).** Hybrid search fusing `$vectorSearch` and `$search` in one aggregation (MongoDB 8.0+). <https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/>
11. **MongoDB Hybrid Search Guide.** Combining full-text and vector search for retrieval. <https://www.mongodb.com/docs/vector-search/hybrid-search/>
