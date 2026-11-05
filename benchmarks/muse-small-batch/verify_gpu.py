#!/usr/bin/env python3
"""Offline module comparison; GPU results are kernel tests, never serving validation."""
import argparse
import ast
import hashlib
import json
import linecache
import statistics
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REF = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
REL = "python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py"
TOKENS = (0, 1, 7, 8, 16, 31, 32, 33)
# Two output-format epsilons for BF16/FP16; FP32 allows reduction/FMA reordering.
TOL = {"bfloat16": (0.015625, 0.015625), "float16": (0.001953125, 0.001953125),
       "float32": (3e-5, 3e-6)}  # (rtol, atol); these are NOT serving tolerances.
KERNEL_NAME = "_post_expert_norm_weighted_sum_kernel"


class KernelLaunchCounter:
    """Forward each launch to the real Triton kernel, counting completed calls."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def counted(*args, **kwargs):
            result = launch(*args, **kwargs)
            self.launches += 1
            return result

        return counted


def checked_kernel_call(fn, arguments, expected_launches):
    """Check dispatch without replacing computation; never use while benchmarking."""
    namespace = fn.__globals__
    if KERNEL_NAME not in namespace:
        raise AssertionError("Kernel dispatch requires a fully loaded module")
    original = namespace[KERNEL_NAME]
    counter = KernelLaunchCounter(original)
    namespace[KERNEL_NAME] = counter
    try:
        result = fn(**arguments)
    finally:
        namespace[KERNEL_NAME] = original
    if counter.launches != expected_launches:
        raise AssertionError(
            f"Expected {expected_launches} kernel launches, observed {counter.launches}"
        )
    return result, counter.launches

def module_from_source(name, source, filename, cpu_only=False, torch=None):
    if cpu_only:
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if
                     isinstance(n, ast.FunctionDef) and n.name == "fused_post_expert_norm_weighted_sum" or
                     isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_MIN_TOKENS_FOR_FUSED" for t in n.targets)]
        if len(tree.body) != 2:
            raise ValueError("Expected exactly one wrapper and one threshold constant")
        namespace = {"torch": torch, "Optional": Optional}
        exec(compile(tree, filename, "exec"), namespace)
        return namespace["fused_post_expert_norm_weighted_sum"]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sglang"):
            raise ValueError("Harness refuses package imports: " + node.module)
        if isinstance(node, ast.Import) and any(a.name.startswith("sglang") for a in node.names):
            raise ValueError("Harness refuses importing the full sglang package")
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    # Triton JIT uses inspect.getsource; retain immutable snapshots without temp files.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), mod.__dict__)
    return mod.fused_post_expert_norm_weighted_sum

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=HERE / "sglang")
    parser.add_argument("--candidate", type=Path, help="Default: target module inside checkout")
    parser.add_argument("--device", choices=("all", "cpu", "cuda"), default="all")
    parser.add_argument("--cpu-only", action="store_true", help="AST-extracted CPU fallback test; requires Torch but no Triton")
    parser.add_argument("--output", type=Path, help="Optional JSON report; otherwise stdout only")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if args.cpu_only:
        args.device = "cpu"
    if args.samples < 3 or args.replays < 1:
        parser.error("--samples must be >=3 and --replays >=1")
    report = {"baseline_ref": REF, "gpu_status": "NOT_RUN_GPU", "results": [], "benchmarks": [],
              "tolerances_rtol_atol": TOL, "scope": "Synthetic isolated kernel, no model/TP8/serving validation"}

    def finish(code):
        rows = report["results"]
        report["passed"] = sum(r["status"] == "PASS" for r in rows)
        report["failed"] = sum(r["status"] == "FAIL" for r in rows)
        report["status"] = "FAIL" if report["failed"] or code == 1 else report["gpu_status"]
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return code
    try:
        checkout = args.checkout.resolve()
        source = subprocess.check_output(["git", "-C", str(checkout), "show", f"{REF}:{REL}"])
        candidate_path = (args.candidate or checkout / REL).resolve()
        candidate = candidate_path.read_bytes()
        report.update(baseline_sha256=hashlib.sha256(source).hexdigest(),
                      candidate_sha256=hashlib.sha256(candidate).hexdigest(), candidate_path=str(candidate_path))
        print("Loading PyTorch...", file=sys.stderr, flush=True)
        import torch
        torch.set_num_threads(1)
        if not args.cpu_only:
            print("Loading Triton...", file=sys.stderr, flush=True)
            import triton
        report.update(torch=torch.__version__, triton=None if args.cpu_only else triton.__version__, cuda_runtime=torch.version.cuda,
                      hip_runtime=torch.version.hip)
        report["load_mode"] = "cpu_ast_extraction" if args.cpu_only else "isolated_full_module"
        baseline = module_from_source("_muse_baseline", source.decode("utf-8"), str(HERE / "_baseline_virtual.py"), args.cpu_only, torch)
        proposed = module_from_source("_muse_candidate", candidate.decode("utf-8"), str(candidate_path), args.cpu_only, torch)
    except ImportError as exc:
        report["blocked_reason"] = f"Missing dependency: {exc}; nothing installed, no tests run"
        return finish(2)
    except Exception as exc:
        report["error"] = f"Setup failed: {type(exc).__name__}: {exc}"
        return finish(1)
    def make(device, dtype, t, h=64, k=8, mode="normal"):
        x = torch.randn(t, k, h, device=device, dtype=dtype)
        scores = torch.rand(t, k, device=device, dtype=torch.float32) / k
        ids = torch.randint(-1, 16, (t, k), device=device, dtype=torch.int32)
        weight = None if mode == "no_weight" else torch.randn(h, device=device, dtype=dtype)
        if mode in ("poison", "all_invalid"):
            if mode == "all_invalid":
                ids.fill_(-1)
            elif t:
                ids[:, 0] = -1
            x[ids < 0] = float("nan")
        if mode.startswith("strided"):
            def strided(a):
                shape = list(a.shape)
                shape[-1] *= 2
                result = torch.empty(shape, dtype=a.dtype, device=a.device)[..., ::2]
                return result.copy_(a)
            if mode == "strided_weight":
                weight = strided(weight)
            elif mode == "strided_ids":
                ids = strided(ids)
            else:
                scores = strided(scores)
        if mode == "no_ids": ids = None
        return dict(expert_output=x, topk_scores=scores, norm_weight=weight, eps=1e-8,
                    weight_first=mode != "weight_after", apply_norm=mode != "no_norm", slot_expert_ids=ids)

    def oracle(a):
        x = a["expert_output"].double()
        if a["slot_expert_ids"] is not None: x = torch.where((a["slot_expert_ids"] >= 0).unsqueeze(-1), x, 0.0)
        w = a["norm_weight"]
        w = w.double() if w is not None else None
        if a["apply_norm"]:
            if w is not None and a["weight_first"]:
                x = x * w
            x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + a["eps"])
            if w is not None and not a["weight_first"]:
                x = x * w
        return (x * a["topk_scores"].double().unsqueeze(-1)).sum(1).to(a["expert_output"].dtype)

    def compare(actual, expected, dtype):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError("Output shape/dtype mismatch")
        if not torch.isfinite(actual).all():
            raise AssertionError("Non-finite output")
        rtol, atol = TOL[str(dtype).split(".")[-1]]
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        delta = (actual.double() - expected.double())
        return {"max_abs": delta.abs().max().item() if delta.numel() else 0.0,
                "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-30)).item()}

    def run_case(label, operation):
        row = {"case": label}
        try:
            row.update(operation() or {}, status="PASS")
        except Exception as exc:
            row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        report["results"].append(row)

    def check(a, dtype, all_invalid=False, expected_launches=None):
        launches = None
        with torch.no_grad():
            if expected_launches is None:
                old, new = baseline(**a), proposed(**a)
            else:
                old, old_count = checked_kernel_call(baseline, a, expected_launches[0])
                new, new_count = checked_kernel_call(proposed, a, expected_launches[1])
                launches = {"baseline": old_count, "candidate": new_count}
            ref = oracle(a)
            if a["expert_output"].device.type == "cpu":
                torch.testing.assert_close(new, old, rtol=0, atol=0)
            metrics = compare(new, old, dtype)
            compare(old, ref, dtype)
            compare(new, ref, dtype)
            if all_invalid and (torch.count_nonzero(old) or torch.count_nonzero(new)):
                raise AssertionError("All-invalid rows must be exactly zero")
            if launches is not None:
                metrics["kernel_launches"] = launches
            return metrics

    torch.manual_seed(20261105)
    cuda_ok = torch.cuda.is_available() and torch.version.hip is None
    devices = (["cpu"] if args.device != "cuda" else []) + (["cuda"] if cuda_ok and args.device != "cpu" else [])
    if "cuda" in devices:
        props = torch.cuda.get_device_properties(0)
        report["gpu_status"] = "PASS_GPU"
        report["device"] = {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
                            "memory_bytes": props.total_memory}
    else:
        report["gpu_reason"] = "Explicit CPU mode" if args.device == "cpu" else "NVIDIA CUDA unavailable (HIP excluded)"
    for device in devices:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            print(f"Checking {device}/{dtype}...", file=sys.stderr, flush=True)
            for t in TOKENS:
                if device == "cpu" and t >= 32:
                    continue  # Baseline itself uses Triton at >=32; no claimed CPU coverage there.
                for mode in ("normal", "no_weight", "no_ids", "weight_after", "no_norm", "poison", "all_invalid"):
                    label = f"{device}/{dtype}/{t}x8x64/{mode}"
                    # Empty batches never launch; every other synthetic CUDA input here is eligible.
                    launches = (int(t >= 32), int(t > 0)) if device == "cuda" else None
                    run_case(label, lambda d=device, dt=dtype, n=t, m=mode, counts=launches:
                             check(make(d, dt, n, mode=m), dt, m == "all_invalid", counts))
            for h in (63, 65, 4096, 4097):
                launches = (0, 1) if device == "cuda" else None
                run_case(f"{device}/{dtype}/7x3x{h}/nearby", lambda h=h, counts=launches:
                         check(make(device, dtype, 7, h, 3), dtype, expected_launches=counts))
            for mode in ("strided_weight", "strided_ids", "strided_scores"):
                # The kernel wrapper copies scores; strided weights/IDs must keep the fallback.
                launches = (0, int(mode == "strided_scores")) if device == "cuda" else None
                run_case(f"{device}/{dtype}/7x8x65/{mode}", lambda m=mode, counts=launches:
                         check(make(device, dtype, 7, 65, mode=m), dtype, expected_launches=counts))
    if "cpu" in devices:
        def gradients():
            a = make("cpu", torch.float32, 7, 65, 3)
            def evaluate(fn):
                b = {k: v.detach().clone().requires_grad_(True) if k in ("expert_output", "topk_scores", "norm_weight") else v for k, v in a.items()}
                out = fn(**b)
                return torch.autograd.grad(out.square().sum(), [b[k] for k in ("expert_output", "topk_scores", "norm_weight")])
            for old, new in zip(evaluate(baseline), evaluate(proposed)):
                compare(new, old, torch.float32)
        run_case("cpu/float32/7x3x65/gradients", gradients)

    if "cuda" in devices:
        def cuda_gradients(required):
            a = make("cuda", torch.float32, 7, 65, 3)
            differentiable = ("expert_output", "topk_scores", "norm_weight")

            def evaluate(fn):
                b = {k: v.detach().clone().requires_grad_(k in required)
                     if k in differentiable else v for k, v in a.items()}
                with torch.enable_grad():
                    output, launches = checked_kernel_call(fn, b, 0)
                    gradients = torch.autograd.grad(
                        output.square().sum(), [b[k] for k in required]
                    )
                return b, output.detach(), gradients, launches

            old_args, old, old_grad, old_count = evaluate(baseline)
            new_args, new, new_grad, new_count = evaluate(proposed)
            compare(new, old, torch.float32)
            for actual, expected in zip(new_grad, old_grad):
                compare(actual, expected, torch.float32)
            # The same tensors still require gradients. no_grad must permit the fused path.
            with torch.no_grad():
                old_inference, old_inference_count = checked_kernel_call(baseline, old_args, 0)
                new_inference, new_inference_count = checked_kernel_call(proposed, new_args, 1)
                compare(new_inference, old_inference, torch.float32)
                compare(new_inference, oracle(a), torch.float32)
            return {"gradient_inputs": list(required),
                    "autograd_kernel_launches": {"baseline": old_count, "candidate": new_count},
                    "no_grad_kernel_launches": {
                        "baseline": old_inference_count, "candidate": new_inference_count}}

        gradient_inputs = ("expert_output", "topk_scores", "norm_weight")
        for required in [(key,) for key in gradient_inputs] + [gradient_inputs]:
            run_case("cuda/float32/7x3x65/gradients/" + "+".join(required),
                     lambda required=required: cuda_gradients(required))

    def capture(fn, a):
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(4): fn(**a)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn(**a)
        return graph, output

    if "cuda" in devices:
        for t in (1, 8, 32):
            print(f"Checking graphs and timing {t}x8x4096...", file=sys.stderr, flush=True)
            a = make("cuda", torch.bfloat16, t, 4096)
            def graphs_and_bench(a=a, t=t):
                old_graph, old_out = capture(baseline, a)
                new_graph, new_out = capture(proposed, a)
                for _ in range(3):
                    a["expert_output"].normal_()  # Fresh values, same pointers, no recapture.
                    old_graph.replay()
                    new_graph.replay()
                    compare(new_out, old_out, torch.bfloat16)
                    compare(new_out, oracle(a), torch.bfloat16)
                if args.skip_benchmark:
                    return
                samples = {"baseline": [], "candidate": []}
                for i in range(args.samples):
                    pair = [("baseline", old_graph), ("candidate", new_graph)]
                    for name, graph in (pair if i % 2 == 0 else pair[::-1]):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.replays):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / args.replays)
                medians = {k: statistics.median(v) for k, v in samples.items()}
                report["benchmarks"].append({"shape": [t, 8, 4096], "dtype": "bfloat16", "median_us": medians,
                    "kind": "unchanged_path_control" if t >= 32 else "candidate_path",
                    "samples_us": samples, "replays_per_sample": args.replays,
                    "scope": "Warm-cache graph replay timing includes host launch pacing; no cold-L2 or serving policy"})
            run_case(f"cuda/bfloat16/{t}x8x4096/graph", graphs_and_bench)
    if any(r["status"] == "FAIL" for r in report["results"]):
        if "cuda" in devices:
            report["gpu_status"] = "FAIL_GPU_OR_CPU"
        return finish(1)
    return finish(0 if cuda_ok or args.device == "cpu" else 2)

if __name__ == "__main__":
    raise SystemExit(main())
