# Tokenization

> **One-liner:** Tokenization converts raw text into a sequence of numbers (token IDs) that a model can process, using a learned vocabulary of subword pieces.

## The Intuition

You know from the Transformers and Attention pages that a transformer processes a sequence of vectors -- one vector per position. But the model starts from raw text, a string of characters. Tokenization is the bridge: it converts that string into a numbered sequence the model can actually work with.

The challenge is harder than it looks.

### Why not just use characters?

The simplest approach: assign a number to every character. "a" = 1, "b" = 2, "cat" = [3, 1, 20]. Done.

The problem: meaning lives at the word level, not the character level. The letters c, a, t in isolation mean almost nothing. A model that processes individual characters needs to reconstruct word-level meaning from scratch every time, requiring far more layers and far more data. The sequence also becomes very long -- 10 characters = 10 positions, burning through context window fast.

```
CHARACTER-LEVEL (problematic)

  "The cat sat."

  "T" "h" "e" " " "c" "a" "t" " " "s" "a" "t" "."
  [84][104][101][32][99][97][116][32][115][97][116][46]

  12 tokens. Each carries almost no semantic content on its own.
  The model must learn that [99,97,116] = "cat" every time.
```

### Why not just use words?

Better: assign a number to every word. "the" = 1, "cat" = 2, "sat" = 3. Common words get their own ID.

The problem: vocabulary explosion and unknown words. English has hundreds of thousands of words, and that ignores technical terms, proper nouns, code, and every other language. New words (slang, product names, scientific terms) appear constantly. A purely word-level vocabulary can never be complete.

```
WORD-LEVEL (also problematic)

  Vocabulary must include:
    "cat", "cats", "catlike", "catfish", "concatenate" ...
    "run", "runs", "running", "runner", "ran", "outruns" ...
    "GPT-4", "MLX", "CUDA", "LTX-Video" ...

  Vocabulary size: 500,000+ words to be reasonable.
  Unknown words become [UNK] -- the model sees nothing.

  "Serendipitous" --> [UNK]  (never seen in training)
  "ChatGPT"       --> [UNK]  (too new)
```

### The subword solution

The practical answer is **subword tokenization**: break text into pieces that are neither characters nor full words, but something in between. Common words stay whole. Rare or novel words get split into recognizable pieces.

```
SUBWORD TOKENIZATION (the actual approach)

  "unhappiness"  -->  ["un", "happiness"]
  "tokenization" -->  ["token", "ization"]
  "serendipitous" --> ["serendip", "itous"]
  "LTX-Video"    -->  ["LTX", "-", "Video"]

  Common words stay whole:
  "the"  --> ["the"]
  "cat"  --> ["cat"]
  "run"  --> ["run"]

  New or rare words decompose into known pieces:
  "GPT-4"  -->  ["G", "PT", "-", "4"]

  Every possible input maps to known token IDs.
  No unknown words -- only known pieces of unknown words.
```

This is why modern models can handle text they've never seen before, including brand new product names, code identifiers, and foreign words -- they decompose them into familiar subparts.


## How It Actually Works

### BPE: building the vocabulary from data

The dominant algorithm for subword tokenization is **Byte-Pair Encoding (BPE)**, introduced for NLP by Sennrich et al. in 2015. It builds the vocabulary from data automatically:

```
BPE ALGORITHM (training the tokenizer)

  Start: vocabulary = every individual character (plus special tokens)
  Input: a large text corpus

  Step 1: Count every adjacent pair of symbols in the corpus.
  Step 2: Find the most frequent pair.
  Step 3: Merge that pair into a new single token. Add to vocabulary.
  Step 4: Replace all occurrences of the pair in the corpus.
  Step 5: Repeat until vocabulary reaches target size.

  Example (simplified):

  Corpus: "low lower lowest"

  Initial tokens: ["l","o","w"," ","e","r","s","t"]

  Iteration 1: Most frequent pair = ("l","o") -> merge to "lo"
               Vocabulary adds: "lo"
               Corpus: "lo w lo wer lo west"

  Iteration 2: Most frequent pair = ("lo","w") -> merge to "low"
               Vocabulary adds: "low"
               Corpus: "low low er low est"

  Iteration 3: Most frequent pair = ("e","r") -> merge to "er"
               ...and so on until vocab size is reached.

  After 50,000 iterations: 50,000-token vocabulary, encoding
  every word in the corpus efficiently with common subwords.
```

The beauty of BPE: the vocabulary size is a hyperparameter. Train it long enough and common English words will be single tokens. Train it more and common phrases become single tokens. Stop early and more words split into pieces. The right stopping point is a trade-off between vocabulary size (memory cost) and sequence length (context cost).

### Vocabulary sizes in practice

```
MODEL VOCABULARY SIZES

  Model           Vocab size    Notes
  --------------- ------------- ---------------------------------
  GPT-2           50,257        BPE, English-heavy
  GPT-4 / GPT-3.5 100,256       tiktoken cl100k_base encoding
  LLaMA-2         32,000        SentencePiece BPE
  LLaMA-3         128,256       tiktoken, much larger multilingual
  CLIP            49,408        BPE, short prompts (77-token limit)
  T5              32,100        SentencePiece, multilingual
  T5-XXL          32,100        Same tokenizer as T5

  Bigger vocab = fewer tokens per sentence = shorter sequences.
  Shorter sequences = less compute per forward pass, more efficient attention.
  Trade-off: bigger embedding table, larger model.
```

### Token IDs: from text to numbers

When you run inference, the tokenizer converts your text string into a list of integers -- token IDs -- that index into the model's embedding table:

```
TOKENIZATION EXAMPLE (GPT-2 tokenizer)

  Text: "Hello world"
  Token IDs: [15496, 995]

  "Hello" --> ID 15496
  "world" --> ID 995

  Text: "Hello, world!"
  Token IDs: [15496, 11, 995, 0]

  "Hello"  --> ID 15496
  ","      --> ID 11
  " world" --> ID 995    (note: space included in token)
  "!"      --> ID 0

  Text: "MLX"
  Token IDs (GPT-2): [44, 25792]  (two pieces, "M" and "LX")

  Text: "The"   vs   " The"
  IDs:  [464]   vs   [383]        (different! leading space matters)
```

The exact mapping is vocabulary-specific. There is no universal standard. "Hello" may be ID 15496 in GPT-2 and a completely different ID in LLaMA.

### The tokenization pipeline end to end

```
TEXT TO MODEL INPUT

  Raw text: "A cat sat on the mat."
                |
         [Tokenizer]
         (vocab lookup + BPE splitting)
                |
         Token IDs: [32, 3797, 3332, 319, 262, 2603, 13]
                |
         [Embedding Table]
         (lookup learned vector for each ID)
                |
         Token embeddings:
           [0.12, -0.34, 0.98, ...]   <- vector for ID 32  ("A")
           [0.55,  0.22, -0.11, ...]  <- vector for ID 3797 (" cat")
           [0.03,  0.87, 0.44, ...]   <- vector for ID 3332 (" sat")
           ...                         shape: [seq_len, d_model]
                |
         [Position Encoding added]
                |
         [Transformer layers]
```

The embedding table is part of the model -- it is learned during training. The tokenizer just tells you which row of that table to look up. The two are inseparable: a given set of token IDs only makes sense with the embedding table that was trained alongside that tokenizer.

### Special tokens

Every tokenizer reserves a few IDs for structural signals that tell the model about sequence boundaries and padding:

```
SPECIAL TOKENS

  [BOS] -- Beginning of Sequence
    Prepended to the start of every input.
    Tells the model "this is the start of a new sequence."
    Common IDs: <s>, <|begin_of_text|>, [CLS]

  [EOS] -- End of Sequence
    Appended at the end (or generated by the model to signal "stop here").
    Tells the model "generation is complete."
    Common IDs: </s>, <|end_of_text|>, [SEP]

  [PAD] -- Padding
    Fills shorter sequences in a batch to a uniform length.
    The model is told to ignore padded positions (via attention mask).
    Critical: the model must not attend to PAD tokens.
    Common IDs: <pad>, [PAD]

  Example -- batching two sequences of different lengths:

    Sequence A: ["The", "cat", "sat"]    --> [464, 3797, 3332]
    Sequence B: ["Hello"]                --> [15496]

    After padding to length 3:
    Sequence A: [464, 3797, 3332]        attention_mask: [1, 1, 1]
    Sequence B: [15496, PAD, PAD]        attention_mask: [1, 0, 0]

    The 0s in the attention mask tell the model to ignore PAD positions.
    Without the mask: PAD tokens contaminate attention, degrading output.
```

Some models also add task-specific special tokens: `[INST]` for instructions in LLaMA chat, `<image>` for multimodal inputs, `[SEP]` between document segments in BERT. These are part of the tokenizer vocabulary and require the model to have been trained with them.

### Context window: the token budget

Every model has a maximum sequence length -- the context window. This is measured in tokens, not words or characters:

```
CONTEXT WINDOW EXAMPLES

  Model            Context window   Approximate word count
  ---------------- ---------------- ----------------------
  GPT-2            1,024 tokens     ~750 words
  LLaMA-2-7B       4,096 tokens     ~3,000 words
  LLaMA-3-8B       8,192 tokens     ~6,000 words
  GPT-4            128,000 tokens   ~96,000 words
  Claude (some)    200,000 tokens   ~150,000 words

  "1 token ≈ 0.75 words" is a rough English heuristic.
  Code, numbers, and non-English text are often less efficient:

  "x = 1234" (8 chars) --> ["x", " =", " 1234"]  = 3 tokens
  "αβγδ" (4 Greek chars) --> ["α","β","γ","δ"]   = 4 tokens (1 each)
  "日本語" (3 chars, Japanese) --> multiple tokens per character

  When the input exceeds the context window, the model truncates it
  (silently, usually) -- input beyond the limit is simply not seen.
```

### SentencePiece and CLIP tokenizers

BPE is not the only algorithm. Two other tokenizers you will encounter in multimodal work:

**SentencePiece** (used by T5, LLaMA): Similar to BPE but works at the byte level. It treats the text as a raw byte sequence rather than a Unicode character sequence, which means it handles any language natively without needing a language-specific pre-processing step. T5 uses SentencePiece with a 32,100-token vocabulary.

**CLIP tokenizer**: Based on BPE with a 49,408-token vocabulary. Has a hard 77-token context limit -- prompts longer than 77 tokens are truncated. This is a critical constraint in many image generation pipelines that use CLIP for text conditioning. For LTX-Video and other advanced video models, T5 is used instead precisely because its 512+ token context handles detailed prompts without truncation.


## Why It Matters for MLX

### Wrong tokenizer = wrong output, guaranteed

The tokenizer and the model are a matched pair. The embedding table in the model was trained on a specific set of token IDs from a specific tokenizer. If you use a different tokenizer -- even one with the same vocabulary size -- the IDs will be wrong and the model will produce garbage:

```
TOKENIZER MISMATCH (catastrophic)

  Correct:
    Text: "a cat sat"
    LLaMA-3 tokenizer --> IDs: [264, 7180, 12700]
    LLaMA-3 model lookup: embedding[264] = (vector for "a")
    Result: correct embedding, model works.

  Wrong tokenizer:
    Text: "a cat sat"
    GPT-2 tokenizer   --> IDs: [64, 3797, 3332]
    LLaMA-3 model lookup: embedding[64] = (vector for something else)
    Result: completely wrong embeddings, model produces nonsense.

  The model has no way to detect this. It just silently
  uses the wrong embeddings. No error, no warning.
  This is one of the most common porting bugs.
```

When porting a model to MLX:
1. Identify which tokenizer the original model uses (check the model card or `config.json` -- look for `tokenizer_class`).
2. Use that exact tokenizer for text input. Do not substitute.
3. If using Hugging Face, `AutoTokenizer.from_pretrained(model_name)` loads the correct tokenizer automatically.

### Tokenizers run on CPU -- do not port them

Tokenizers are fast and they run on CPU. There is no reason to re-implement them in MLX. Use the existing libraries:

```python
# Hugging Face transformers (most common)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
token_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]

# OpenAI tiktoken (GPT-2/GPT-3/GPT-4 tokenizers)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
token_ids = enc.encode("Hello world")

# Once you have token IDs (a list of integers), convert to MLX:
import mlx.core as mx
ids = mx.array(token_ids)  # shape: [seq_len] or [batch, seq_len]
```

The tokenizer produces a plain Python list or NumPy array of integers. Converting that to an MLX array is trivial. The heavy lifting -- BPE merges, vocab lookups -- all happens in the tokenizer library, not in your MLX code.

### LTX-Video: T5 tokenizer in the conditioning pipeline

LTX-Video uses a T5 tokenizer and T5 encoder for text conditioning. Understanding the full pipeline from raw prompt to cross-attention is essential for working on the model:

```
LTX-VIDEO TEXT CONDITIONING PIPELINE

  Text prompt: "a dog running through a field of tall grass at sunset"
                |
         [T5 Tokenizer]          (runs on CPU, SentencePiece BPE)
         (32,100-token vocab,     Not ported to MLX -- use HuggingFace
          up to 512 tokens)
                |
         Token IDs: [3, 9, 1483, 1510, 145, 3, 307, 13, 1954, ...]
                    shape: [1, seq_len]
                |
         [T5 Encoder]             (transformer, runs on GPU/Apple Silicon)
         (encoder-only,           This IS ported to MLX if you want
          12 to 24 layers)        the text encoding to run on-device
                |
         Text embeddings:         shape: [1, seq_len, 1024]  (T5-XL)
                |                        or [1, seq_len, 4096]  (T5-XXL)
                |
         [Stored as K, V in cross-attention]
         (computed once per prompt, reused across all denoising steps)
                |
                +----------> cross-attention in every DiT block
                             Video patches (Q) attend to text (K, V)
```

Key details for LTX-Video:
- T5 tokenizer handles prompts up to 512 tokens (vs. CLIP's 77-token limit).
- The T5 encoder runs once per prompt and produces `encoder_hidden_states`.
- These hidden states are passed as `K` and `V` into every cross-attention layer in the DiT.
- If you change the prompt, you rerun the T5 encoder. If you change the seed but keep the prompt, you reuse the cached hidden states.

### The 77-token cliff in CLIP-based models

For reference when working with other image generation models (Stable Diffusion 1.x, 2.x): CLIP tokenizer truncates at 77 tokens. This is not a soft limit -- tokens beyond 77 are silently dropped:

```
CLIP 77-TOKEN LIMIT (Stable Diffusion)

  Short prompt (safe):
    "a tabby cat"
    Tokens: ["a", " tabby", " cat"]  = 3 tokens. Fine.

  Long prompt (danger zone):
    "A highly detailed photorealistic image of a tabby cat
     sitting on a wooden chair in a sunlit kitchen with
     white walls, potted plants on the windowsill, a cup
     of coffee on the counter, and warm afternoon light..."
    Tokens: ~60 tokens. Approaching the 77-token limit.

  Too-long prompt (silent truncation):
    Any prompt exceeding 77 tokens:
    Token 78 onward is discarded. The model never sees it.
    The truncation is silent -- no error, no warning in most
    implementations.

  LTX-Video uses T5 (512 tokens) to avoid this limitation.
  Detailed video prompts describing actions, lighting, camera
  movement, and scene elements easily exceed 77 tokens.
```

### Debugging tokenization issues

When output quality is poor and you suspect a tokenization problem:

```
TOKENIZATION DEBUGGING CHECKLIST

  1. Verify the tokenizer:
       print(tokenizer.__class__.__name__)
       # Should match what the model card specifies

  2. Inspect token IDs directly:
       ids = tokenizer("test input")["input_ids"]
       print(ids)
       print(tokenizer.convert_ids_to_tokens(ids))
       # Lets you see the actual subword pieces

  3. Check for truncation:
       ids = tokenizer(long_prompt)["input_ids"]
       print(f"Prompt is {len(ids)} tokens")
       print(f"Model max is {tokenizer.model_max_length} tokens")
       # If len(ids) > model_max_length, you are losing content

  4. Check special tokens:
       print(tokenizer.bos_token, tokenizer.bos_token_id)
       print(tokenizer.eos_token, tokenizer.eos_token_id)
       print(tokenizer.pad_token, tokenizer.pad_token_id)
       # Ensure these match what the model expects

  5. Round-trip test:
       ids = tokenizer.encode("Hello world")
       decoded = tokenizer.decode(ids)
       print(decoded)  # Should be "Hello world" (or close to it)
       # Significant differences indicate unusual tokenizer behavior
```


## Key Terms

| Term | Definition |
|------|------------|
| **Tokenization** | The process of converting raw text into a sequence of discrete tokens (integers) that a model can process |
| **Token** | A single unit in the tokenizer's vocabulary; may be a character, a subword piece, a whole word, or a special symbol |
| **Tokenizer** | The component that performs tokenization; maps strings to token ID sequences and back; always paired with a specific model's vocabulary |
| **Token ID** | The integer index of a token in the vocabulary; what gets passed to the model's embedding table |
| **Vocabulary** | The fixed set of all tokens a tokenizer knows; tokens outside the vocabulary are split into smaller known pieces |
| **BPE (Byte-Pair Encoding)** | The dominant subword tokenization algorithm; starts with characters, iteratively merges the most frequent adjacent pairs until the target vocabulary size is reached |
| **Subword** | A token that represents part of a word (e.g., "##tion", "un##", "##ness"); allows the vocabulary to cover rare and unknown words by decomposing them into known pieces |
| **Embedding** | A learned vector representation for each token ID; stored in the model's embedding table; converts a token ID into a dense vector that the transformer processes |
| **Special Tokens** | Reserved token IDs that carry structural signals rather than semantic content: BOS (beginning of sequence), EOS (end of sequence), PAD (padding), SEP (separator), UNK (unknown) |
| **BOS (Beginning of Sequence)** | Special token prepended to indicate the start of a new input sequence; tells the model where the text begins |
| **EOS (End of Sequence)** | Special token indicating end of input or output; the model generates this token to signal it has finished |
| **PAD (Padding)** | Special token used to fill shorter sequences in a batch to a uniform length; attention mask is set to 0 at PAD positions so the model ignores them |
| **Context Window** | The maximum number of tokens a model can process in a single forward pass; determined at training time; inputs longer than this limit are truncated |
| **SentencePiece** | A tokenization library from Google that implements BPE at the byte level; handles any language natively; used by T5, LLaMA, and others |
| **CLIP Tokenizer** | A BPE tokenizer with 49,408-token vocabulary and a hard 77-token context limit; used in Stable Diffusion 1.x and 2.x for text conditioning |
| **T5 Tokenizer** | A SentencePiece tokenizer with 32,100 tokens; handles up to 512 tokens; used by LTX-Video for text conditioning via T5 encoder |


## Sources

- Sennrich, R., Haddow, B., & Birch, A. (2015). "Neural Machine Translation of Rare Words with Subword Units." *ACL 2016*. Introduces BPE for NLP tokenization -- the algorithm underlying GPT-2, LLaMA, and most modern tokenizers. Available at [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
- Hugging Face. "Tokenizers Documentation." Covers BPE, WordPiece, and SentencePiece; includes `AutoTokenizer`, `PreTrainedTokenizer`, and tokenizer training. Available at [huggingface.co/docs/transformers/tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- OpenAI tiktoken. Fast BPE tokenizer used by GPT-2 through GPT-4; good reference for seeing how byte-level BPE differs from character-level BPE. Available at [github.com/openai/tiktoken](https://github.com/openai/tiktoken)
- Karpathy, A. "Let's build the GPT tokenizer." A video walkthrough of implementing BPE from scratch, with clear intuition for the merge algorithm and why tokenization is trickier than it looks. Available at [youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE)


## See Also

- [Neural Networks](02-neural-networks.md) -- prerequisite; models operate on numbers, not text; tokenization is how text becomes numbers
- [Transformers](05-transformers.md) -- the architecture that consumes token IDs; the embedding table lookup is the first operation in every transformer forward pass
- [Attention](06-attention.md) -- operates on the token embeddings produced by the embedding table; the KV-cache stores keys and values per token, so token count directly determines cache size
