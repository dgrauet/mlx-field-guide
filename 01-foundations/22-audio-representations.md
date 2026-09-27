# Audio Representations

> **One-liner:** Audio reaches a model in one of three forms -- a raw **waveform**, a **mel spectrogram** (a picture of which frequencies are loud over time), or **codec tokens** (discrete codes from a neural audio codec) -- and each is produced by preprocessing code with a dozen parameters that must match the training setup exactly, because the model can't tell you when they don't.

## The Intuition

A microphone records air pressure thousands of times per second. That list of numbers, the **waveform**, is complete but awkward for a model: one second of speech is 16,000-48,000 values, and the information that matters (pitch, timbre, phonemes) is spread across them as oscillations.

```
THREE WAYS AUDIO ENTERS A MODEL

  waveform          [T samples]              16,000 values per second (16 kHz)
     |
     |  STFT + mel filters + log
     v
  mel spectrogram   [n_mels, frames]         80 x 100 per second (Whisper)
                                               -> speech recognition, vocoder input
     |
  (or) neural codec encoder + quantizer
     v
  codec tokens      [codebooks, frames]      8 x 12.5 per second (Mimi)
                                               -> LLM-style audio generation
```

The mel spectrogram is to audio what an image is to a scene: a fixed, hand-designed representation that a model then learns from. Codec tokens are the audio equivalent of text tokens, which is why modern text-to-speech and speech-to-speech models can reuse LLM machinery.

---

## How It Actually Works

### Waveform and sample rate

The **sample rate** is how many values per second: 16 kHz for speech recognition (Whisper), 24 kHz for most TTS and codecs, 44.1/48 kHz for music. A model trained at one rate interprets audio at another as sped up or slowed down, so input must be **resampled** to the model's rate first. Values are floats in [-1, 1]. Whisper's loader converts 16-bit PCM by dividing by 32768.

### From waveform to spectrogram: the STFT

The **short-time Fourier transform (STFT)** cuts the waveform into overlapping windows and takes the Fourier transform of each, giving the strength of each frequency in each time slice:

```
  n_fft      window length in samples        Whisper: 400  (25 ms at 16 kHz)
  hop        step between windows             Whisper: 160  (10 ms -> 100 frames/s)
  window     taper applied to each slice      Hann (periodic)
  center     pad n_fft/2 on both sides so      true (reflect padding)
               frame t is centered on sample t*hop
  power      |X| (magnitude) or |X|² (power)  2

  output: (n_fft/2 + 1) frequency bins x frames   ->  201 x 3000 for 30 s of Whisper audio
```

### The mel scale and the log

Human hearing resolves low frequencies finely and high ones coarsely. A **mel filterbank** -- 80 or 128 overlapping triangular filters, narrow at low frequencies and wide at high ones -- pools the 201 frequency bins into 80 **mel bands**. Then a log compresses the huge dynamic range, the same way we perceive loudness.

Whisper's full recipe, from `whisper/audio.py`:

```
  power spectrogram (drop the last frame)
  -> mel = filters @ power            filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
  -> log10(max(mel, 1e-10))
  -> max(log_spec, log_spec.max() - 8)    # 80 dB dynamic range
  -> (log_spec + 4) / 4                    # roughly [-1, 1.5]
```

Implemented in MLX 0.32.2 with `mx.fft.rfft` over framed windows, it matches the PyTorch reference to 1e-4. Each deviation from the recipe, measured on 5 s of synthetic audio (the output spans about -0.4 to 1.4):

```
  deviation                                          max |diff|   mean |diff|
  -------------------------------------------------  ----------   -----------
  everything matched                                    0.0001        0.0000
  symmetric window (mx.hanning) instead of periodic     0.033         0.0007
  magnitude instead of power                            0.52          0.17
  natural log instead of log10                          0.60          0.33
  no center padding (also 498 frames instead of 500)    0.95          0.09
  HTK mel filters instead of Slaney (librosa default)   1.86          0.48
```

The window subtlety deserves a note: `mx.hanning(N)` (like `np.hanning`) is **symmetric**, while `torch.hann_window(N)` is **periodic** by default. For N = 4: `[0, 0.75, 0.75, 0]` vs `[0, 0.5, 1, 0.5]`. Build the periodic window as `np.hanning(N + 1)[:-1]`. The error is small, but it's the kind that breaks a parity test.

### From spectrogram back to audio: vocoders

A spectrogram discards phase, so it can't be inverted exactly. A **vocoder** is a network trained to generate a waveform from a mel spectrogram. HiFi-GAN is the classic one: a stack of transposed convolutions that upsample by the hop length (see [Convolutions](19-convolutions.md)). Older TTS systems generate mel spectrograms and hand them to a vocoder. A vocoder only works with mels made by exactly the recipe it was trained on (same sample rate, n_fft, hop, n_mels, fmin/fmax, log type).

### Neural audio codecs and residual vector quantization

A **neural audio codec** (SoundStream, EnCodec, DAC, Mimi, SNAC) is an autoencoder with a discrete bottleneck. The encoder downsamples the waveform to a low frame rate. Each frame vector is then replaced by entries from learned **codebooks**, whose entries are [centroids](../glossary.md#centroid), exactly as in codebook quantization. The decoder turns the codes back into a waveform.

One codebook can't capture a frame precisely. **Residual vector quantization (RVQ)** stacks them: codebook 1 quantizes the vector, codebook 2 quantizes what codebook 1 missed, and so on. Demonstrated in MLX (random 16-dim vectors, k-means codebooks of 256 entries):

```
  codebooks   bits per vector   relative error
      1              8              0.736
      2             16              0.542
      4             32              0.295
      8             64              0.089
```

Each codebook costs the same bits and removes a roughly constant fraction of the remaining error. That is why codecs expose bitrate as "number of codebooks", and why models often generate the first codebook (the coarse content) differently from the rest.

```
  codec (config)        sample rate   frame rate   codebook size   codebooks used
  EnCodec 24 kHz          24 kHz        75 Hz          1024         2-32 (1.5-24 kbps)
  DAC 44 kHz              44.1 kHz      86 Hz          1024          9
  Mimi (Moshi)            24 kHz        12.5 Hz        2048          8 (first one semantic)
  SNAC 24 kHz             24 kHz        12/23/47 Hz    4096          3 scales
```

Frame rate × codebooks is the number of tokens per second a generator must produce: 12.5 × 8 = 100 for Mimi, 75 × 32 = 2,400 for EnCodec at 24 kbps. Low frame rates are what make real-time speech models possible.

---

## Why It Matters for MLX

### Preprocessing is part of the model

For audio, the preprocessing is as much a part of the model as the weights. A Whisper port with the wrong mel filters runs and transcribes -- badly -- with no error anywhere. Port the feature extraction with the same care as a layer, and test it on its own: compute the reference's features for a fixed clip, compute yours, and compare numerically before running the model.

MLX has what's needed: `mx.fft.rfft` for the STFT, `mx.pad(mode="reflect")` for centering (MLX 0.32+), and the mel filterbank can be precomputed once with librosa and shipped as a constant, which is what Whisper itself does.

### Port failures

```
AUDIO PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Transcription garbage or hallucinated      mel mismatch: HTK vs Slaney filters,
                                               log vs log10, power vs magnitude,
                                               missing normalization
  Output audio sped up / chipmunk /          sample rate mismatch: input not
    slowed down                                resampled, or output written at the
                                               wrong rate
  Small parity error in features only        symmetric vs periodic window,
                                               or center padding mode
  Off-by-one frame count, timestamps drift   center padding missing, or last
                                               frame not dropped
  Metallic / buzzy vocoder output            mel recipe differs from the one the
                                               vocoder was trained on
  Codec output noisy, right length           codebooks applied in the wrong order,
                                               or residuals not summed
  Codec output wrong length                  frame rate / hop mismatch between
                                               encoder and decoder configs
  Clicks at chunk boundaries (streaming)     chunks decoded independently without
                                               the codec's overlap or state
```

---

## Key Terms

| Term | Definition |
|------|------------|
| **Waveform** | The raw audio signal: amplitude samples in [-1, 1] at a fixed sample rate. |
| **Sample rate** | Samples per second (16 kHz for Whisper, 24 kHz for most TTS); input must match the model's. |
| **STFT** | Short-time Fourier transform: windowed Fourier transforms over time; defined by n_fft, hop, window, centering. |
| **Hop length** | Samples between consecutive STFT frames; sets the frame rate (160 at 16 kHz = 100 frames/s). |
| **Mel spectrogram** | STFT power pooled into perceptual mel bands, usually log-compressed; the standard input to speech models. |
| **Mel filterbank** | The triangular filters that map frequency bins to mel bands; Slaney (librosa default) and HTK variants differ. |
| **Vocoder** | Network that generates a waveform from a mel spectrogram (e.g. HiFi-GAN). |
| **Neural audio codec** | Autoencoder with a discrete bottleneck that turns audio into tokens and back (EnCodec, DAC, Mimi, SNAC). |
| **Residual vector quantization (RVQ)** | Stacked codebooks, each quantizing the residual error of the previous ones. |
| **Codebook** | A learned table of vectors (centroids); each code is an index into it. |

---

## Sources

- Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper). [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356); preprocessing in `whisper/audio.py`: [github.com/openai/whisper](https://github.com/openai/whisper/blob/main/whisper/audio.py)
- Kong, J., Kim, J., & Bae, J. (2020). "HiFi-GAN." [arxiv.org/abs/2010.05646](https://arxiv.org/abs/2010.05646)
- Zeghidour, N., et al. (2021). "SoundStream: An End-to-End Neural Audio Codec" (RVQ). [arxiv.org/abs/2107.03312](https://arxiv.org/abs/2107.03312)
- Défossez, A., et al. (2022). "High Fidelity Neural Audio Compression" (EnCodec). [arxiv.org/abs/2210.13438](https://arxiv.org/abs/2210.13438)
- Kumar, R., et al. (2023). "High-Fidelity Audio Compression with Improved RVQGAN" (DAC). [arxiv.org/abs/2306.06546](https://arxiv.org/abs/2306.06546)
- Défossez, A., et al. (2024). "Moshi: a speech-text foundation model for real-time dialogue" (Mimi codec). [arxiv.org/abs/2410.00037](https://arxiv.org/abs/2410.00037)
- Codec configs on the Hugging Face Hub: `facebook/encodec_24khz`, `descript/dac_44khz`, `kyutai/mimi`, `hubertsiuzdak/snac_24khz`.
- librosa `filters.mel` (Slaney vs HTK): [librosa.org/doc/0.11.0/generated/librosa.filters.mel.html](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html)
- Measurements on this page: MLX 0.32.2 log-mel against the Whisper recipe in PyTorch on synthetic audio; RVQ demonstration in MLX 0.32.2 with random data.

---

## See Also

- [Audio](../02-ecosystem/06-audio.md) -- the audio model ecosystem (Whisper, TTS, music) on CUDA and MLX
- [Tokenization](09-tokenization.md) -- text tokens, the counterpart of codec tokens
- [Quantization](10-quantization.md) and [Embeddings & RAG](../02-ecosystem/10-embeddings-rag.md) -- codebooks and centroids in other contexts
- [Convolutions & Patchification](19-convolutions.md) -- the 1D and transposed convolutions inside codecs and vocoders
- [Numerical Stability](12-numerical-stability.md) -- why the log is clamped at 1e-10
