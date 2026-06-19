# Agents & Tool Use: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- full agent frameworks, grammar-constrained decoding, production tool routers) | 🟡 MLX (Growing -- the model-side primitives exist; the high-level libraries that make them turnkey are thin)

## What This Domain Covers

Making an [LLM](03-llms.md) *do* things rather than just talk: call a function, query a database, run a search, then read the result and decide what to do next. This is **tool use** (also "function calling"), and an **agent** is the loop around it -- a model that plans, calls tools, observes outcomes, and iterates until a task is done.

Two capabilities make this reliable, and they are really the same problem seen from two angles:

- **Tool use / function calling** -- the model emits a structured request ("call `get_weather` with `{city: "Paris"}`"), your code runs the function, and the result is fed back into the conversation.
- **Constrained / structured generation** -- forcing the model's output to conform to a schema (valid JSON, a regex, one of a fixed set of choices). Without it, a model "mostly" emits valid JSON and then occasionally doesn't -- and an agent that parses tool calls from free text breaks on that occasional malformed output.

Constrained generation is what turns "the model usually produces parseable tool calls" into "the model *always* does." That is why these two topics belong on one page.

```
THE AGENT LOOP

  user goal
     |
     v
  [LLM] --plan--> emit tool call (structured)   <- constrained generation
     |                  keeps this parseable
     |                  v
     |             your code runs the tool
     |                  |
     |             tool result
     |                  |
     +<-----------------+   (feed result back, repeat)
     |
     v
  final answer once the goal is met
```

---

## The CUDA Ecosystem

### Tool Calling

Tool calling is driven by two things: a model **trained** to emit tool calls (Llama 3.1+, Qwen 2.5, Mistral, the Hermes fine-tunes, Command R), and a **chat template** that knows how to format tool definitions into the prompt. In the Hugging Face stack, `tokenizer.apply_chat_template(messages, tools=[...])` renders the tools into the model's expected format, and serving engines (vLLM, TGI) expose this through the OpenAI `tools` / `tool_choice` API, returning parsed `tool_calls` objects.

### Constrained Generation

This is where the CUDA ecosystem is genuinely ahead. Several mature libraries guarantee schema-valid output by masking the model's logits at each step so only tokens that keep the output valid can be sampled:

```
CONSTRAINED-DECODING LIBRARIES (CUDA-native)

  Library            Constrains output to
  -------            --------------------
  Outlines           regex, JSON schema, Pydantic models, choices
  XGrammar           context-free grammars; very fast, vLLM-integrated
  lm-format-enforcer JSON schema / regex; works across backends
  guidance           templated control flow + constraints

  Mechanism (all of them):
    at each step, compute which next tokens keep the output valid,
    set every other token's logit to -inf, then sample as usual.
    The model literally cannot emit invalid structure.
```

These plug into vLLM and Transformers and are standard equipment for production agents.

### Agent Frameworks

**LangChain**, **LlamaIndex**, **smolagents** (Hugging Face), **CrewAI**, and **AutoGen** provide the loop, tool registries, memory, and multi-agent orchestration. Critically, these are **pure Python orchestration** -- like the RAG stack, they don't care what runs the model underneath, only that it can be called.

---

## The MLX Ecosystem

### Tool Calling

mlx-lm uses the model's Hugging Face tokenizer and its chat template directly. For a tool-trained model whose template supports it, you pass tool definitions the same way you would anywhere in the HF ecosystem:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]
prompt = tokenizer.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, tokenize=False
)  # -> a string; generate() re-encodes it
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
# When calling generate() directly, `response` is the model's tool-call TEXT —
# YOU parse it (e.g. json.loads the function-call block) and dispatch.
```

> Pass `tokenize=False`. `load()` returns a `TokenizerWrapper`, so omitting it happens to work; but a raw Hugging Face tokenizer with the default `tokenize=True` returns a `BatchEncoding` that `generate` cannot consume. Asking for the string and letting `generate` re-encode is the robust idiom (it is what mlx-lm's own CLI does).

Two paths diverge here. If you call `generate()` **directly in Python**, mlx-lm formats the input tools but hands you the raw output to parse yourself. If you go through **`mlx_lm.server`** (the OpenAI-compatible endpoint), it actually parses the model's tool-call output into structured `tool_calls` via a per-model tool parser and returns them in the OpenAI shape -- so existing OpenAI clients get structured calls back. What the server does *not* expose is `tool_choice`-style *forcing* of a specific call.

### Constrained Generation

mlx-lm exposes the right primitive: `generate` and `stream_generate` accept `sampler` and `logits_processors` keyword arguments, and `mlx_lm.sample_utils` ships standard ones. A **logits processor** is a callable that receives the tokens generated so far and the next-token logits and returns modified logits -- exactly the hook the CUDA libraries build on:

```python
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Minimal example: restrict generation to a fixed set of allowed token ids.
allowed = set(tokenizer.encode("yes no"))   # toy whitelist

def constrain_to_allowed(tokens, logits):
    mask = mx.full(logits.shape, -mx.inf)
    for tid in allowed:
        mask[..., tid] = 0.0
    return logits + mask

response = generate(
    model, tokenizer, prompt=prompt, max_tokens=10,
    logits_processors=[constrain_to_allowed],
)
```

For production use you would vectorize the mask rather than loop in Python (mlx-lm's own `sample_utils` uses the functional `logits.at[..., idx].add(...)` form); the loop above is just the minimal illustration of the hook.

What's missing is the **layer above** this primitive: there is no first-class, widely-adopted MLX integration of a schema/grammar library. Outlines -- the de facto CUDA tool -- does **not** list MLX among its backends (it targets Transformers, vLLM, Ollama, and hosted APIs). So on MLX today you either hand-write logits processors for your schema, or generate-then-validate-then-retry. The hook is there; the ergonomic library on top is the gap.

### Agent Frameworks

The orchestration frameworks are pure Python and run on macOS, but -- exactly as with [RAG](10-embeddings-rag.md) -- their default model backends assume PyTorch/CUDA or a hosted API. To drive an agent with a local MLX model you wrap mlx-lm as a custom LLM, or point a framework at an OpenAI-compatible base URL served by `mlx_lm.server`. Functional, but glue you write rather than `pip install`.

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Tool-call prompt formatting** | 🟢 `apply_chat_template(tools=...)` | 🟢 Same HF chat template, works identically | None for tool-trained models |
| **Tool-call output parsing** | 🟢 vLLM/TGI return structured `tool_calls` | 🟡 Direct `generate()`: you parse the text. `mlx_lm.server`: parsed into `tool_calls` for you | Small -- covered via the server; manual on the raw API |
| **Server `tools` API** | 🟢 OpenAI-compatible, incl. `tool_choice` forcing | 🟡 `mlx_lm.server` accepts `tools` and returns `tool_calls`; no `tool_choice` forcing | Small -- forcing a specific call is the missing piece |
| **Constrained decoding primitive** | 🟢 logit masking everywhere | 🟢 `logits_processors` in `generate` | None -- the hook exists |
| **Schema/grammar library** | 🟢 Outlines, XGrammar, lm-format-enforcer | 🔴 No first-class, widely-adopted MLX integration | Large -- the biggest gap on this page |
| **Agent frameworks** | 🟢 LangChain/LlamaIndex/smolagents native | 🟡 Work via custom LLM wrapper or server URL | Small -- glue code, not new infra |

---

## What the Gap Actually Costs You

If you are building **a single-agent tool-using app locally**: modest cost. Tool-call *input* formatting is identical to anywhere else. Go through `mlx_lm.server` and tool-call *output* is parsed into `tool_calls` for you; call `generate()` directly and you write a small parse-and-dispatch function plus a retry-on-invalid-JSON guard. A Qwen 2.5 or Llama 3.1 model on MLX runs the loop fully offline.

If you depend on **guaranteed schema-valid output** (strict JSON for downstream systems, grammar-constrained DSLs): this is the real friction. You either hand-roll logits processors or accept generate-then-validate retries, where CUDA users get `Outlines`/`XGrammar` for free. For high-stakes structured output this can be the deciding factor.

If you run **a multi-agent orchestration** with LangChain/CrewAI: the frameworks run fine; you supply the MLX backend wrapper. No turnkey integration, but no blocker either.

---

## Contributing Here

This page describes one of the clearest "primitive exists, library doesn't" gaps in the MLX ecosystem -- high-leverage because the hard part (the decoding hook) is already done:

1. **A schema-constrained generation library for mlx-lm.** Build JSON-schema / regex constraints on top of `logits_processors` (the XGrammar or lm-format-enforcer approach, ported). The single highest-impact item on this page.
2. **`tool_choice` forcing and broader tool-parser coverage in `mlx_lm.server`.** The server already parses tool calls into OpenAI-style `tool_calls`; the open work is forcing a specific call (`tool_choice`) and extending the per-model parsers to more architectures.
3. **First-class MLX backends for LangChain / LlamaIndex / smolagents.** Turn "wrap it yourself" into `pip install`, mirroring the same opportunity called out for [RAG](10-embeddings-rag.md).

See [Open Opportunities](../03-contributing/03-open-opportunities.md) for how these rank against other gaps.

---

## Sources

- mlx-lm (`sampler` / `logits_processors`, `mlx_lm.sample_utils`): [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Outlines: [github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
- XGrammar: [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
- lm-format-enforcer: [github.com/noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)
- guidance: [github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)
- LangChain: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- LlamaIndex: [github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)
- smolagents: [github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)
- Hugging Face tool-use chat templates: [huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/main/en/chat_templating)

---

## See Also

- [LLMs](03-llms.md) -- the base inference layer; tool use and constrained generation are capabilities layered on top of mlx-lm's `generate`
- [Embeddings & RAG](10-embeddings-rag.md) -- the sibling pattern; agents often call a retriever as one of their tools, and the "primitive works, library is glue" story is identical
- [Vision-Language Models](11-vision-language-models.md) -- multimodal agents that can also see; same tool-loop, image-capable model
- [Serving & Deployment](09-serving-deployment.md) -- `mlx_lm.server` is how you expose an agent backend over an OpenAI-compatible URL
- [Tokenization](../01-foundations/09-tokenization.md) -- chat templates and special tokens are how tool definitions enter the prompt; constrained decoding operates on token ids
