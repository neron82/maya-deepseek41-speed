# DeepSeek-V4.1-Flash IQ2_XXS, two GPUs with a GPU 0 expert helper

Measured 2026-10-29 on one PC: AMD Ryzen 7 3700X (8 cores, 16 threads), 128 GiB RAM, two NVIDIA RTX 3090 24 GB
cards. The fast card is physical GPU 1 on PCIe 4 x16 and runs the layers and the native vision tower; physical GPU 0
on the slower PCIe link is a 24 GB expert helper. Measured links were about 26.3 GB/s to GPU 1 and 2.86 GB/s to GPU 0,
with no peer access. Linux, CUDA only.

Model: DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS (2.00 bpw). Engine: `maya-ds4`, measured build
SHA-256 `d9e133a6c473f8b06cb35b150b843c8a4dc8c5f60630716faebc2baff6b2db34`; the 2026-10-29 build
(`88433596c7d236d0d066e4c0fbb50e123900a2945b6696eb0b9886abb25d0e4c`) recorded 21.36 / 25.20 tok/s for the same pair.
Rebuilding this branch produces a fresh binary with its own hash.
All six routed experts stay enabled; the quant and the 524288-token allocation are unchanged; DSpark and suffix
drafting are off.

## Method

The same preserved 8130-token review prompt and 512 greedy tokens (temperature 0, finish reason `length`) were sent
twice to one running server. The first request is cold after load; the second reuses 8129 prompt tokens. Startup is
excluded from the throughput numbers. This is a serving run, not a fixed-output-token microbenchmark: free-generated
continuations can differ, and run-to-run timing moves.

| Request | Prompt tokens | Generated | Prefill | Decode | Wall |
|---|---:|---:|---:|---:|---:|
| Cold | 8,130 | 512 | 522.1 tok/s (15.57 s) | 21.35 tok/s (23.98 s) | 39.56 s |
| Repeat, 8,129 reused | 8,130 | 512 | 57.4 ms | 25.82 tok/s (19.83 s) | 19.89 s |

The cold run's layer card held 1,763 expert slots and left 13,597 experts outside it, of which 10,858 were in the host
RAM tier. The preset configured a 24 GB helper pool with a 98 GB host warm cap.

## Controls on the same prompt

| Configuration | Prefill | Cold decode | Repeat decode | Cold wall |
|---|---:|---:|---:|---:|
| Maya fused, GPU 1 only (no helper) | 506.2 tok/s | 15.40 tok/s | 19.48 tok/s | 49.31 s |
| Strata optimized, GPU 1 only (separate engine) | 413.5 tok/s | 14.87 tok/s | 17.97 tok/s | — |

The single-card Maya row isolates the helper's effect on this engine. The Strata row is a different implementation,
shown only as an external control; it is not the configuration this repository packages and not a like-for-like fork
comparison.

## Reproducing the preset

[`helper-config.json`](helper-config.json) is the portable copy of the measured env and CLI: `CUDA_VISIBLE_DEVICES=1,0`
(fast physical GPU 1 first), `DS4_HELPER=1`, `DS4_HELPER_GB=24`, `DS4_WARM_CAP_GB=98`, `--gpus 0 --tp 1`, plus the
kernel/copy switches the run used. The helper device is the second visible device and is deliberately not part of
`--gpus`; this is not tensor parallelism. Every local path in the file is a `<placeholder>` to replace, and the fast
card's identity must be verified on the target machine before reusing the `1,0` order. Start it with:

```sh
python -m serve.server --engine strata --config bench/results/deepseek41-gpu-helper/helper-config.json
```

## Limits

- One cold and one repeat request per configuration, one machine and one thermal state; not a sweep.
- The helper sums selected experts on GPU 0 instead of the CPU lane. The engine's documentation notes that this can
  change the last bits of an answer; no bit-identity or full-model quality claim is made.
- Linux and CUDA were measured. Windows and AMD helper paths are untested upstream.
- Repeat prefill is small only because 8129 prompt tokens were reused from the cold request.

## Files and provenance

[`results.json`](results.json) holds the sanitized scalars, the controls and each source artifact's SHA-256;
[`manifest.json`](manifest.json) records the same provenance and the sanitisation rules. The original artifacts remain
the primary record. Prompts, token ids, generated text, request bodies, monitor streams, logs, GPU UUIDs and absolute
paths are not published.
