# Muse small-batch operator demonstration

The recorded GPU checks passed for this isolated operator. This is not a serving benchmark.

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0; 7.96 GiB)
- PyTorch 2.13.0+cu130; Triton 3.7.1; CUDA runtime 13.0
- Checks: 344 passed (148 CPU, 196 CUDA), 0 failed
- Baseline revision: `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`
- Baseline source SHA-256: `9b816e971817f96a22d2d09424dc8dfa6f6020e18de73fb895674c76494fe591`
- Candidate source SHA-256: `a1d75e436702a28b0c38e16bbbc054581cd2e9c9a6737b958bd6ab39a3983ed8`

## Measurements

Times are microseconds per warm-cache CUDA Graph replay, measured with CUDA events. They include host launch pacing and are not pure kernel execution times. The baseline/candidate ratio is a ratio of medians: above 1 favors the candidate; below 1 favors the baseline. Every recorded configuration is shown.

| Shape (tokens × top-k × hidden) | Type | Role | Baseline median µs (min–max) | Candidate median µs (min–max) | Baseline/candidate | Samples per variant | Replays/sample |
|---|---|---|---:|---:|---:|---:|---:|
| 1 × 8 × 4096 | bfloat16 | Candidate path | 21.9416 (20.5066–31.3819) | 11.0531 (9.17984–19.2186) | 1.985× | 21 | 200 |
| 8 × 8 × 4096 | bfloat16 | Candidate path | 32.9878 (32.7482–44.7062) | 9.73008 (8.6624–17.8654) | 3.39× | 21 | 200 |
| 32 × 8 × 4096 | bfloat16 | Unchanged-path control | 11.2938 (8.59712–22.8008) | 10.9352 (8.54816–18.8622) | 1.033× | 21 | 200 |

[Raw evidence with all timing samples and case results](results.json)

[Reproduction instructions](REPRODUCE.md) · [Recorded environment](environment.json) · [Candidate patch](candidate.patch)

## Numerical checks and limits

The harness compares the candidate with the original implementation and an FP64 reference. Its unit-test tolerances (rtol, atol) are:

- bfloat16: (0.015625, 0.015625)
- float16: (0.001953125, 0.001953125)
- float32: (3e-05, 3e-06)

These tolerances are not the competition's acceptance criteria. Results on an RTX 5060, or any other local GPU named above, do not establish H200 performance, full-model numerical correctness, TP8 behavior, or end-to-end serving speed. This demonstration neither uses the official evaluator nor constitutes a competition submission. Timing variability is shown rather than treated as a confidence interval.
