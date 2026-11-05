# Reproducing the operator demonstration

This branch contains a demonstration of the existing experimental change in
`python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py`.
The candidate is also published separately on `experiment/muse-small-batch`.
Use that separate branch for a competition patch; the demonstration files are not
part of the candidate patch.

## Environment

The recorded run uses Windows, CPython 3.14.5, PyTorch 2.13.0+cu130,
triton-windows 3.7.1.post27, and an NVIDIA RTX 5060 Laptop GPU.
The Triton module reports its upstream version as 3.7.1.
No model weights, competition requests, private datasets, or API credentials are used.

In a dedicated Python 3.14 environment, the corresponding packages are:

```text
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install triton-windows==3.7.1.post27
```

Other supported CUDA environments can run the same test, but their timings are
separate measurements. Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
and the [Triton Windows compatibility table](https://github.com/triton-lang/triton-windows).

## Run from the repository root

If using a shallow clone, fetch the reference commit first:

```text
git fetch --depth 1 origin a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
python benchmarks/muse-small-batch/verify_gpu.py --checkout . --device all --samples 21 --replays 200 --output demo-results.raw.json
python benchmarks/muse-small-batch/summarize_demo.py demo-results.raw.json --output benchmarks/muse-small-batch/README.md --public-json benchmarks/muse-small-batch/results.json
```

The harness loads the original source directly from the fixed Git revision and the
candidate from the checkout. It does not import the complete SGLang package.
The random seed is fixed and the numerical tolerances are recorded in the result.
The summary command refuses a successful-demo summary if the GPU checks fail or
if the required CUDA cases or measured benchmarks are missing.

## Interpretation

- Numerical checks compare against the original implementation and an FP64 reference.
- Instrumented checks observe real Triton launches and verify the fallback and autograd paths.
- Instrumentation is removed before timing.
- Timings use CUDA events around repeated CUDA Graph replays, alternate the order of
  the two variants, and include the host's launch pacing.
- The 32-token configuration is an unchanged-path control: both versions use Triton.
- Every recorded timing sample and tested configuration is published in `results.json`.

This is an AI-assisted, synthetic operator demonstration. It does not establish
H200 performance, TP8 correctness, full-model output agreement, or end-to-end serving
speed. It is not a submission to the competition or an upstream merge request.
