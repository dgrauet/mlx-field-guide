# Audio: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- production-ready speech, TTS, and music generation) | 🟡 MLX (Emerging -- Whisper and growing TTS coverage; music generation gap remains)

## What This Domain Covers

Three distinct sub-domains of audio generation and processing:

- **Speech-to-text (STT / ASR):** Transcribing spoken audio to text. Whisper and its derivatives dominate.
- **Text-to-speech (TTS):** Synthesizing natural-sounding speech from text. Encompasses voice cloning, multilingual synthesis, and expressive prosody.
- **Music generation:** Creating music from text descriptions or conditioning audio. A newer and rapidly evolving area.

These three share underlying deep learning infrastructure (transformers, diffusion, VAEs) but have distinct model architectures, training pipelines, and deployment requirements. The CUDA ecosystem is mature across all three. The MLX ecosystem is strongest in STT, growing in TTS, and essentially absent for music generation.

---

## The CUDA Ecosystem

### Speech-to-Text: Whisper and its Derivatives

OpenAI's Whisper (2022) is the canonical open-source speech recognition model. A single encoder-decoder transformer trained on 680,000 hours of web audio, it supports 99 languages, handles accented speech, background noise, and mixed-language audio better than most commercial ASR systems.

```
WHISPER ARCHITECTURE

  Audio input (raw waveform or mel spectrogram)
      |
      +--> Log-mel spectrogram: [T_audio] -> [80 mel bands, T/2 frames]
      |    (25ms windows, 10ms hop, 80 mel filters)
      |
      +--> Audio Encoder (transformer encoder)
      |    Input:  [80, T/2] mel spectrogram
      |    Output: [T/2, d_model]  -- encoded audio representations
      |
      +--> Text Decoder (transformer decoder, autoregressive)
           Attends to encoder output via cross-attention
           Generates tokens: <|transcribe|> text tokens <|endoftext|>
           Or for translation: <|translate|> [English tokens] <|endoftext|>

  VARIANTS (by size):
    tiny:    39M params   ~1GB   fast; lower accuracy
    base:    74M params   ~1GB   good balance
    small:   244M params  ~2GB   strong accuracy
    medium:  769M params  ~3GB   very accurate
    large-v2: 1.5B params ~3GB   best open-source accuracy (long used)
    large-v3: 1.5B params ~3GB   improved (2023)
    turbo:   809M params  ~1.6GB large-v3 accuracy at small speed
```

**Whisper in Python (CUDA):**

```python
import whisper

model = whisper.load_model("large-v3")  # downloads ~3GB weights

result = model.transcribe(
    "interview.mp3",
    language="en",        # optional; detected automatically if omitted
    task="transcribe",    # or "translate" to convert to English
    fp16=True,            # use half precision for speed
    word_timestamps=True  # include per-word timestamps
)

print(result["text"])
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
```

**faster-whisper** is a reimplementation of Whisper using CTranslate2, a C++ inference engine with [INT8](../glossary.md#int8) quantization. It runs significantly faster than the original PyTorch Whisper on [CPU](../glossary.md#cpu) and CUDA:

```
FASTER-WHISPER vs ORIGINAL WHISPER (approximate)

  Model       Hardware      Original Whisper    faster-whisper (INT8)
  -----       --------      ----------------    ---------------------
  large-v3    RTX 4090      ~25 RTF*           ~60-80 RTF (INT8)
  large-v3    CPU (M1)      ~2 RTF             ~8-12 RTF
  
  * RTF = Real-Time Factor (1.0 = real-time; higher is faster)
  
  faster-whisper advantages:
    - INT8 quantization: 4x memory reduction, 2-4x speed
    - Batched transcription: process multiple audio segments in parallel
    - Word-level timestamps via CTC alignment
    - VAD (Voice Activity Detection) integration
```

**WhisperX** extends faster-whisper with forced alignment for word-level timestamps and speaker diarization (identifying who is speaking when). It is the tool of choice for subtitle generation and meeting transcription.

### Text-to-Speech

CUDA's TTS ecosystem spans multiple generations of architecture, from pure neural vocoders to fully end-to-end transformer models.

**Bark (Suno AI)** is a fully generative TTS model that can produce not just speech but music, background sounds, and non-verbal cues (laughter, sighs). It uses a GPT-style language model at its core:

```
BARK ARCHITECTURE

  Text input
      |
      +--> Semantic tokenizer (coarse, text->semantic tokens)
      |    Uses a text encoder to convert to semantic token sequence
      |
      +--> Coarse acoustic model (GPT-style, autoregressive)
      |    Converts semantic tokens -> coarse acoustic tokens
      |
      +--> Fine acoustic model
      |    Upsamples coarse -> fine acoustic tokens
      |
      +--> Codec decoder (EnCodec)
           Converts acoustic tokens -> waveform

  Supports: voice cloning (provide 3-10s audio prompt),
            non-speech sounds, multilingual output
  Limitation: slow (GPT-style autoregressive); not real-time
```

**XTTS (Coqui TTS)** is a multilingual voice cloning model. Given a 3-6 second reference audio clip, it can synthesize new speech in the same voice in multiple languages:

```python
from TTS.api import TTS

# XTTS-v2: voice cloning, multilingual
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

tts.tts_to_file(
    text="This is a test of voice cloning.",
    speaker_wav="reference_voice.wav",    # 3-6 second reference clip
    language="en",
    file_path="output.wav"
)
```

**F5-TTS** (2024) is a flow-matching based TTS model. Using the same architectural approach as LTX-Video (flow matching, DiT), it achieves state-of-the-art naturalness with zero-shot voice cloning.

```
F5-TTS vs XTTS

  Metric           XTTS-v2              F5-TTS
  ------           -------              ------
  Architecture     Transformer + VITS   DiT + Flow Matching
  Voice cloning    3-6s reference       ~10s reference
  Languages        17 languages         English + Chinese primary
  Quality          High                 State-of-the-art (2024)
  Speed            ~5-10 RTF (CUDA)     ~8-12 RTF (CUDA)
  MLX support      Partial              mlx-audio (growing)
```

**Piper** is a fast, lightweight neural TTS engine optimized for low-latency CPU inference. It uses a VITS-based architecture and can run in real-time on a Raspberry Pi. The trade-off: voice quality is lower than Bark or XTTS, but latency is extremely low (sub-100ms for short phrases).

**Kokoro** is a newer lightweight TTS model (2024) that achieves high quality at small model size (~82M parameters). It has become popular for local deployment due to its balance of quality and speed.

### Music Generation

Music generation on CUDA has advanced rapidly since Meta's AudioCraft release:

**MusicGen (Meta / AudioCraft)** generates music from text descriptions. It is an autoregressive language model that generates EnCodec audio tokens:

```
MUSICGEN ARCHITECTURE

  Text input:      "upbeat jazz with piano and bass, 120 BPM"
      |
      +--> T5 text encoder (conditioning)
      |
      +--> Transformer decoder (autoregressive, 4 codebooks in parallel)
      |    Generates EnCodec tokens at 50 tokens/second
      |    4 parallel streams (4 EnCodec codebooks)
      |
      +--> EnCodec decoder
           Converts tokens -> waveform at 32kHz

  VARIANTS:
    musicgen-small:  300M params    ~1.5 min/min on RTX 4090
    musicgen-medium: 1.5B params    ~3 min/min on RTX 4090
    musicgen-large:  3.3B params    ~6 min/min on RTX 4090
    musicgen-stereo: stereo output variant

  OUTPUT: Up to 30 seconds of audio (longer with continuation)
  MLX status: No port.
```

**Stable Audio (Stability AI)** uses a latent diffusion approach for music generation, similar to image diffusion models. It generates higher quality audio than autoregressive models for music, at the cost of longer generation times.

**Suno and Udio** are commercial music generation services (not open source). They represent the current quality ceiling for AI music, but are not locally runnable.

**AudioCraft** is Meta's open-source audio generation suite. In addition to MusicGen, it includes:
- **AudioGen**: generates sound effects from text descriptions
- **EnCodec**: the neural audio codec used as the token backbone for both MusicGen and AudioGen

---

## The MLX Ecosystem

### mlx-audio

mlx-audio is the primary MLX library for audio tasks. It provides:
- Whisper port for speech recognition
- Growing TTS model support (Kokoro, CSM, others in progress)
- Audio utility functions for loading, processing, and saving audio

```
MLX-AUDIO COVERAGE (Q2 2025)

  Speech-to-text:
    Whisper (all variants: tiny through large-v3, turbo)
    Status: Functional and well-maintained

  Text-to-speech:
    Kokoro: lightweight TTS, MLX port available
    CSM (Cartesia Speech Model): voice cloning TTS
    Orpheus-TTS: expressiveness-focused TTS
    F5-TTS: community port in progress
    XTTS: not yet ported

  Music generation:
    MusicGen: not ported
    Stable Audio: not ported
    AudioCraft: not ported

  Audio utilities:
    Audio loading (via soundfile / librosa)
    Mel spectrogram computation
    EnCodec codec (partial; needed for Bark/MusicGen)
```

**Whisper in MLX:**

```python
import mlx_whisper

# Load and transcribe
result = mlx_whisper.transcribe(
    "interview.mp3",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
)

print(result["text"])
```

The MLX Whisper implementation matches the original in output quality. Performance on Apple Silicon is competitive with faster-whisper on CPU, though slower than a CUDA [GPU](../glossary.md#gpu):

```
WHISPER PERFORMANCE COMPARISON (large-v3, approximate RTF)

  Hardware              Framework           RTF (higher = faster than real-time)
  --------              ---------           ------------------------------------
  NVIDIA RTX 4090       faster-whisper      ~60-80 RTF (INT8)
  NVIDIA RTX 4090       original Whisper    ~25 RTF (FP16)
  Apple M3 Max          mlx-whisper         ~15-20 RTF
  Apple M2 Ultra        mlx-whisper         ~20-25 RTF
  Apple M4 Max          mlx-whisper         ~20-30 RTF
  Apple M3 Max          whisper.cpp (Metal) ~12-18 RTF
  Intel CPU (i9)        faster-whisper INT8 ~4-6 RTF

  CONCLUSION: MLX Whisper on Apple Silicon is 3-5x slower than
  faster-whisper on CUDA, but 3-5x faster than CPU inference.
  For transcription tasks (offline, not real-time), this is
  practical -- a 1-hour recording takes 3-4 minutes on M3 Max.
```

### Whisper.cpp with Metal

Whisper.cpp (Georgi Gerganov, same author as llama.cpp) is a C++ port of Whisper with [Metal](../glossary.md#metal) backend support. It does not use MLX -- it uses Metal Shading Language directly for GPU compute. But it is worth noting for Apple Silicon users because it provides extremely fast Whisper inference:

```
WHISPER.CPP vs MLX-WHISPER (Apple Silicon)

  Framework         Backend    large-v3 RTF    Notes
  ---------         -------    ------------    -----
  mlx-whisper       MLX        ~15-20 RTF      Python API; easy integration
  whisper.cpp       Metal      ~20-30 RTF      C++ binary; faster; no Python overhead
  faster-whisper    CPU        ~3-5 RTF        CPU only; slower on Apple Silicon
  original Whisper  CPU/MPS    ~2-4 RTF        PyTorch MPS; slowest

  Practical choice:
    Python integration needed: use mlx-whisper
    Maximum speed, CLI usage: use whisper.cpp with Metal
    Both work well for offline transcription on Apple Silicon
```

### TTS on MLX: Kokoro and CSM

**Kokoro** on MLX is the most complete TTS option for Apple Silicon:

```python
# Kokoro TTS via mlx-audio (example pattern)
from mlx_audio.tts.models.kokoro import KokoroModel

model = KokoroModel.from_pretrained("mlx-community/kokoro-tts-mlx")

audio = model.generate(
    text="Welcome to the MLX field guide.",
    voice="af_heart",   # voice style selection
    speed=1.0
)

# audio: mx.array [samples], float32, 24kHz
```

**CSM (Cartesia Speech [Model](../glossary.md#model))** on MLX provides voice cloning capability:

```
CSM CAPABILITIES (MLX port)

  Voice cloning:   YES -- provide reference audio clip
  Languages:       English primary
  Quality:         High; natural prosody
  Speed:           Slower than real-time on M1/M2; near real-time on M3 Max+
  Status:          Available in mlx-audio; growing community usage
```

### The mlx-community Audio Models

Similar to LLMs, the mlx-community organization on Hugging Face hosts converted audio models:
- `mlx-community/whisper-large-v3-mlx`
- `mlx-community/whisper-turbo-mlx`
- `mlx-community/kokoro-tts-mlx`
- `mlx-community/csm-1b-mlx`

The conversion and upload pipeline is similar to LLMs: models converted from PyTorch weights using MLX conversion scripts and uploaded with standardized naming.

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Whisper STT** | 🟢 Original Whisper (Python), faster-whisper (INT8), WhisperX (diarization) | 🟢 mlx-whisper (all model sizes); competitive performance | Small -- MLX Whisper is well-covered; gap is in speed and advanced features (diarization) |
| **Whisper real-time transcription** | 🟢 faster-whisper + VAD: real-time streaming transcription; <500ms latency | 🟡 Possible with chunked audio; not as optimized | Medium -- real-time streaming requires careful implementation; not turnkey in mlx-audio |
| **Speaker diarization (who spoke when)** | 🟢 WhisperX + pyannote.audio; production-quality diarization | 🔴 No MLX diarization equivalent | Large -- diarization requires a separate speaker embedding model (pyannote); not ported to MLX |
| **VAD (Voice Activity Detection)** | 🟢 Silero VAD, WebRTC VAD integrated with faster-whisper | 🟡 Basic; not deeply integrated with mlx-whisper | Medium -- VAD is important for streaming/long audio; integration exists but not polished |
| **Bark (generative TTS)** | 🟢 Full Bark: speech, music, non-verbal sounds; voice cloning via audio prompt | 🔴 Not ported to MLX | Large -- Bark's expressiveness (laughter, sighs, music) has no MLX equivalent |
| **XTTS voice cloning** | 🟢 XTTS-v2: 17-language voice cloning; 3-second reference audio | 🔴 Not ported to MLX | Large -- multilingual voice cloning is a common production use case |
| **F5-TTS** | 🟢 Full port; state-of-the-art quality; flow matching | 🟡 Community port in progress in mlx-audio | Medium -- in active development; likely to close soon |
| **Kokoro TTS** | 🟢 Full PyTorch version | 🟢 mlx-audio has Kokoro; functional | Small -- both sides have it; MLX port may be slightly behind latest version |
| **CSM voice cloning** | 🟢 PyTorch reference implementation | 🟡 mlx-audio has CSM; growing | Small -- CSM is available on MLX; English-focused |
| **Piper (fast CPU TTS)** | 🟢 Extremely fast; 10+ languages; sub-100ms latency | 🔴 No MLX port; uses ONNX/C++ runtime | Small -- Piper is CPU-optimized; MLX would not improve it significantly; runs fine on CPU |
| **MusicGen (text-to-music)** | 🟢 AudioCraft; small/medium/large; stereo; continuation | 🔴 No MLX port | Large -- music generation is entirely absent from MLX ecosystem |
| **Stable Audio** | 🟢 Latent diffusion music; high quality | 🔴 No MLX port | Large -- no MLX music generation via diffusion |
| **AudioGen (sound effects)** | 🟢 AudioCraft AudioGen; text-to-sound-effect | 🔴 No MLX port | Medium -- sound effect generation useful for video; no MLX equivalent |
| **EnCodec neural codec** | 🟢 PyTorch; backbone for MusicGen, Bark, AudioGen | 🟡 Partial in mlx-audio; needed for Bark/MusicGen port | Medium -- EnCodec port is a prerequisite for porting Bark and MusicGen |
| **TTS real-time streaming** | 🟢 Piper and Coqui TTS support streaming synthesis | 🟡 Not robustly implemented in mlx-audio | Medium -- streaming TTS (produce audio as text is processed) is important for chatbot applications |
| **STT performance** | 🟢 RTX 4090 + faster-whisper: 60-80 RTF; real-time for any use case | 🟡 M3 Max + mlx-whisper: 15-20 RTF; fast for offline; marginal for real-time on M1 | Medium -- MLX is fast enough for most transcription; falls behind on real-time streaming |
| **Multilingual TTS** | 🟢 XTTS (17 langs), Bark (multiple), Piper (many languages) | 🟡 Kokoro (English+); CSM (English); limited multilingual | Large -- multilingual TTS is a significant gap; especially important for non-English use cases |
| **TTS fine-tuning / voice training** | 🟢 Coqui TTS training scripts; custom voice training on CUDA | 🔴 No MLX TTS fine-tuning support | Large -- training a custom TTS voice on MLX is not currently feasible |

---

## Performance Deep Dive

### STT Performance

```
WHISPER LARGE-V3 PERFORMANCE (1-hour audio -> transcript, approximate)

  Hardware              Framework           Wall time
  --------              ---------           ---------
  NVIDIA RTX 4090       faster-whisper      ~45-60s total
  NVIDIA RTX 4090       original Whisper    ~2.5-3 min
  Apple M3 Max          mlx-whisper         ~3-5 min
  Apple M2 Ultra        mlx-whisper         ~2.5-4 min
  Apple M4 Max          mlx-whisper         ~2-3.5 min
  Apple M1 (16GB)       mlx-whisper         ~8-12 min
  Intel i9 (CPU)        faster-whisper INT8 ~10-15 min

  PRACTICAL TAKEAWAY:
    For offline transcription of meetings, podcasts, or interviews,
    MLX Whisper on M3 Max is workable -- a 1-hour recording takes
    ~4 minutes. CUDA is 4-5x faster but both are "fast enough"
    for non-real-time use.

    For real-time transcription (live meeting capture, live captions),
    CUDA + faster-whisper is clearly better. MLX on M3 Max is near
    real-time for smaller models (turbo, small) but not large-v3.
```

### TTS Performance

```
TTS PERFORMANCE (1 minute of synthesized audio, approximate)

  Hardware              Model             Generation time
  --------              -----             ---------------
  NVIDIA RTX 4090       Bark (medium)     ~15-30s
  NVIDIA RTX 4090       XTTS-v2           ~8-15s
  NVIDIA RTX 4090       F5-TTS            ~5-10s
  NVIDIA RTX 4090       Kokoro            ~2-5s
  Apple M3 Max          Kokoro (MLX)      ~5-15s
  Apple M3 Max          CSM (MLX)         ~10-25s
  Apple M3 Max          Bark (no port)    N/A
  Apple M3 Max          XTTS (no port)    N/A

  PRACTICAL TAKEAWAY:
    For Kokoro (available on MLX), Apple Silicon is usable but
    2-3x slower than CUDA. For voice cloning workflows (Bark, XTTS),
    there is no MLX option -- the models are not ported.
```

---

## Contribution Opportunities

**Bark MLX port.** Bark's architecture (semantic tokenizer -> coarse acoustic model -> fine acoustic model -> EnCodec decoder) consists entirely of transformer components that map naturally to MLX. The coarse and fine acoustic models are GPT-style autoregressive transformers; EnCodec is a 1D convolutional codec. Porting Bark would give MLX users the most expressive TTS available in the open-source ecosystem -- voice cloning, non-verbal sounds, background music -- in a single model. Requires porting EnCodec first (see below).

**EnCodec MLX port.** EnCodec is the neural audio codec that underpins Bark, MusicGen, and AudioGen. It is a prerequisite for porting all three. The architecture is a 1D convolutional encoder-decoder with residual vector quantization (RVQ). Porting EnCodec to MLX is a bounded, self-contained contribution that unlocks a cascade of downstream ports.

**MusicGen MLX port.** After EnCodec, MusicGen is the next most impactful audio contribution. The text-to-music model uses a transformer decoder over EnCodec tokens -- similar in structure to LLM token generation. The T5 text encoder is already available via transformers; the MusicGen transformer itself maps to mlx.nn.TransformerDecoder. This would be the first music generation capability on Apple Silicon via MLX.

**XTTS multilingual voice cloning.** XTTS-v2 is a practical voice cloning tool used in real applications (audiobooks, content localization, accessibility tools). Its architecture (VITS-based transformer + speaker embeddings + multilingual decoder) is more complex than Bark or MusicGen but is well-documented. A port would fill the largest single TTS gap in the MLX ecosystem.

**WhisperX diarization.** Speaker diarization -- identifying who spoke each segment of a transcript -- requires a speaker embedding model (typically pyannote.audio's speaker diarization pipeline). Porting pyannote's ECAPA-TDNN or similar speaker embedding model to MLX, and integrating it with mlx-whisper, would bring full WhisperX-equivalent functionality to Apple Silicon.

**Real-time streaming STT.** Implementing a streaming transcription pipeline on top of mlx-whisper: chunk incoming audio into overlapping windows, transcribe each chunk, merge results with attention to boundary artifacts. This is an engineering integration task more than a modeling task -- mlx-whisper itself is ready; the streaming wrapper is what's missing. Useful for live captions, meeting transcription, and voice assistants.

**Silero VAD MLX port.** Voice Activity Detection (VAD) is a small but important preprocessing step for all STT pipelines: it identifies which parts of an audio stream contain speech, reducing wasted computation on silence. Silero VAD is a small LSTM model. Porting it to MLX would enable clean real-time streaming pipelines.

---

## Sources

- Whisper (OpenAI): [github.com/openai/whisper](https://github.com/openai/whisper). The canonical speech recognition model; architecture reference and model weights.
- faster-whisper: [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper). CTranslate2 reimplementation; INT8 quantization reference.
- WhisperX: [github.com/m-bain/whisperx](https://github.com/m-bain/whisperx). Word-level timestamps and speaker diarization on top of faster-whisper.
- mlx-audio: [github.com/lucasnewman/mlx-audio](https://github.com/lucasnewman/mlx-audio). The primary MLX library for audio; Whisper, Kokoro, CSM.
- whisper.cpp: [github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp). C++ Whisper with Metal backend; alternative to mlx-whisper for maximum speed.
- Bark (Suno AI): [github.com/suno-ai/bark](https://github.com/suno-ai/bark). Fully generative TTS with voice cloning and non-verbal sounds.
- Coqui TTS / XTTS: [github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS). XTTS-v2 and the broader Coqui TTS ecosystem.
- F5-TTS: Research paper and implementation; flow matching TTS model; state-of-the-art naturalness.
- AudioCraft (Meta): [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft). MusicGen, AudioGen, EnCodec -- the open-source music generation suite.
- Piper TTS: [github.com/rhasspy/piper](https://github.com/rhasspy/piper). Fast, lightweight neural TTS; ONNX runtime; optimized for CPU/embedded.
- Kokoro TTS: Lightweight high-quality TTS model (~82M params); available on Hugging Face.
- Silero VAD: [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad). Voice activity detection model; important for streaming STT pipelines.

---

## See Also

- [Serving & Deployment](09-serving-deployment.md) -- real-time audio serving considerations; streaming and latency constraints
- [Transformers](../01-foundations/05-transformers.md) -- Whisper, Bark, MusicGen, and most TTS models use transformer architectures; the encoder-decoder pattern (Whisper) and decoder-only pattern (Bark, MusicGen) are the two dominant structures
- [Tokenization](../01-foundations/09-tokenization.md) -- audio models tokenize sound just as LLMs tokenize text; EnCodec's residual vector quantization (RVQ) is the audio equivalent of BPE tokenization
- [LLMs](03-llms.md) -- MusicGen and Bark generate audio tokens using GPT-style autoregressive decoding; the same sequence modeling machinery that powers LLMs is used for audio generation
- [Diffusion](../01-foundations/08-diffusion.md) -- Stable Audio uses latent diffusion for music generation; the same diffusion process that generates images applied to a mel spectrogram latent
- [Frameworks](02-frameworks.md) -- PyTorch is the CUDA audio ecosystem's foundation; mlx-audio is the MLX equivalent; the conversion pattern (PyTorch -> numpy -> MLX) applies to audio models exactly as it does to image models
