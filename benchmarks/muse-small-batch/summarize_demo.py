#!/usr/bin/env python3
"""Publish a gated Markdown summary and path-free evidence from verify_gpu.py."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import quote


SCOPE = "Synthetic operator test and warm-cache CUDA Graph replay; not model or serving validation"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def safe_text(value):
    require(isinstance(value, str) and value and len(value) < 160,
            "Missing or invalid environment metadata")
    require(not any(mark in value for mark in ("/", "\\", ":", "\n", "\r", "|", "`")),
            "Environment metadata must not contain paths or Markdown markup")
    return value


def public_report(report):
    """Validate actual rows and timings, then copy only publishable evidence fields."""
    require(isinstance(report, dict), "Expected a JSON object")
    require(report.get("status") == report.get("gpu_status") == "PASS_GPU",
            "GPU validation did not pass; no validated demo will be generated")
    passed, failed = report.get("passed"), report.get("failed")
    require(type(passed) is int and passed >= 195 and type(failed) is int and failed == 0,
            "Need at least 195 passing cases and zero failures")
    rows = report.get("results")
    require(isinstance(rows, list) and len(rows) == passed,
            "Case rows do not match the reported pass count")
    require(all(isinstance(row, dict) and row.get("status") == "PASS" for row in rows),
            "Every case row must pass")
    require(sum(str(row.get("case", "")).startswith("cuda/") for row in rows) >= 195,
            "Need at least 195 CUDA case rows; CPU cases cannot establish GPU validation")
    require(report.get("load_mode") == "isolated_full_module",
            "GPU validation requires the complete module, not CPU AST extraction")
    result = {"scope": SCOPE, "status": "PASS_GPU", "gpu_status": "PASS_GPU",
              "passed": passed, "failed": 0, "load_mode": "isolated_full_module"}
    for field, length in (("baseline_ref", 40), ("baseline_sha256", 64), ("candidate_sha256", 64)):
        value = report.get(field)
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{%d}" % length, value),
                "Missing or invalid revision/hash: " + field)
        result[field] = value.lower()
    for field in ("torch", "triton", "cuda_runtime"):
        result[field] = safe_text(report.get(field))
    require(report.get("hip_runtime") is None, "This demo requires NVIDIA CUDA")
    result["hip_runtime"] = None
    device = report.get("device", {})
    require(isinstance(device, dict), "Missing GPU metadata")
    capability, memory = device.get("capability"), device.get("memory_bytes")
    require(isinstance(capability, list) and len(capability) == 2
            and all(type(v) is int and v >= 0 for v in capability)
            and type(memory) is int and memory > 0, "Invalid GPU capability or memory metadata")
    result["device"] = {"name": safe_text(device.get("name")),
                        "capability": capability, "memory_bytes": memory}
    # Deliberately omit candidate_path, errors, and unknown fields that may expose local data.
    result["results"] = []
    for row in rows:
        case = row.get("case")
        require(isinstance(case, str) and re.fullmatch(r"(?:cpu|cuda)/[A-Za-z0-9_./+=-]+", case),
                "Invalid case label")
        clean = {"case": case, "status": "PASS"}
        for key in ("max_abs", "relative_l2"):
            if key in row:
                require(number(row[key]), "Invalid numerical error metric")
                clean[key] = row[key]
        for key in ("kernel_launches", "autograd_kernel_launches", "no_grad_kernel_launches"):
            if key in row:
                counts = row[key]
                require(isinstance(counts, dict) and set(counts) == {"baseline", "candidate"}
                        and all(type(v) is int and v >= 0 for v in counts.values()),
                        "Invalid kernel launch counts")
                clean[key] = counts.copy()
        if "gradient_inputs" in row:
            values = row["gradient_inputs"]
            require(isinstance(values, list) and all(v in (
                "expert_output", "topk_scores", "norm_weight") for v in values),
                "Invalid gradient input metadata")
            clean["gradient_inputs"] = values[:]
        result["results"].append(clean)
    tolerances = report.get("tolerances_rtol_atol")
    require(isinstance(tolerances, dict) and set(tolerances) == {"float16", "bfloat16", "float32"}
            and all(isinstance(v, (list, tuple)) and len(v) == 2 and all(number(x) for x in v)
                    for v in tolerances.values()), "Missing or invalid numerical tolerances")
    result["tolerances_rtol_atol"] = tolerances
    benches = report.get("benchmarks")
    require(isinstance(benches, list) and len(benches) >= 2, "Need at least two measured benchmarks")
    result["benchmarks"] = []
    seen = set()
    for bench in benches:
        require(isinstance(bench, dict), "Invalid benchmark row")
        shape, dtype = bench.get("shape"), bench.get("dtype")
        require(isinstance(shape, list) and len(shape) == 3
                and all(type(v) is int and v > 0 for v in shape)
                and dtype in ("float16", "bfloat16", "float32"), "Invalid benchmark shape or dtype")
        signature = (tuple(shape), dtype)
        require(signature not in seen, "Duplicate benchmark shapes cannot establish coverage")
        seen.add(signature)
        samples, medians = bench.get("samples_us", {}), bench.get("median_us", {})
        require(isinstance(samples, dict) and isinstance(medians, dict), "Invalid timing metadata")
        for name in ("baseline", "candidate"):
            values = samples.get(name)
            require(isinstance(values, list) and len(values) >= 3
                    and all(number(v, positive=True) for v in values),
                    "Need at least three finite positive timings per variant")
            require(number(medians.get(name), positive=True)
                    and math.isclose(statistics.median(values), medians[name], rel_tol=1e-12),
                    "Reported median does not match the raw timing samples")
        require(len(samples["baseline"]) == len(samples["candidate"]), "Unpaired timing sample counts")
        replays = bench.get("replays_per_sample")
        require(type(replays) is int and replays > 0, "Invalid replay count")
        result["benchmarks"].append({"shape": shape[:], "dtype": dtype,
            "median_us": {key: medians[key] for key in ("baseline", "candidate")},
            "samples_us": {key: samples[key][:] for key in ("baseline", "candidate")},
            "replays_per_sample": replays, "scope": SCOPE})
        if "kind" in bench:
            require(bench["kind"] in ("candidate_path", "unchanged_path_control"),
                    "Invalid benchmark role")
            result["benchmarks"][-1]["kind"] = bench["kind"]
    return result


def summarize(report, raw_filename):
    clean = public_report(report)
    require(isinstance(raw_filename, str) and raw_filename not in ("", ".", "..")
            and "/" not in raw_filename and "\\" not in raw_filename,
            "Raw evidence link must be a filename, not a local path")
    gpu = clean["device"]
    cpu_count = sum(row["case"].startswith("cpu/") for row in clean["results"])
    cuda_count = sum(row["case"].startswith("cuda/") for row in clean["results"])
    lines = ["# Muse small-batch operator demonstration", "",
        "The recorded GPU checks passed for this isolated operator. This is not a serving benchmark.", "",
        f"- GPU: {gpu['name']} (compute capability {'.'.join(map(str, gpu['capability']))}; "
        f"{gpu['memory_bytes'] / 2**30:.2f} GiB)",
        f"- PyTorch {clean['torch']}; Triton {clean['triton']}; CUDA runtime {clean['cuda_runtime']}",
        f"- Checks: {clean['passed']} passed ({cpu_count} CPU, {cuda_count} CUDA), {clean['failed']} failed",
        f"- Baseline revision: `{clean['baseline_ref']}`",
        f"- Baseline source SHA-256: `{clean['baseline_sha256']}`",
        f"- Candidate source SHA-256: `{clean['candidate_sha256']}`", "",
        "## Measurements", "",
        "Times are microseconds per warm-cache CUDA Graph replay, measured with CUDA events. "
        "They include host launch pacing and are not pure kernel execution times. "
        "The baseline/candidate ratio is a ratio of medians: above 1 favors the candidate; "
        "below 1 favors the baseline. Every recorded configuration is shown.", "",
        "| Shape (tokens × top-k × hidden) | Type | Role | Baseline median µs (min–max) | "
        "Candidate median µs (min–max) | Baseline/candidate | Samples per variant | Replays/sample |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for bench in clean["benchmarks"]:
        medians, samples = bench["median_us"], bench["samples_us"]
        times = [f"{medians[name]:.6g} ({min(samples[name]):.6g}–{max(samples[name]):.6g})"
                 for name in ("baseline", "candidate")]
        role = {"candidate_path": "Candidate path", "unchanged_path_control": "Unchanged-path control"}.get(bench.get("kind"), "Measured")
        lines.append(f"| {' × '.join(map(str, bench['shape']))} | {bench['dtype']} | {role} | "
            f"{times[0]} | {times[1]} | {medians['baseline'] / medians['candidate']:.4g}× | "
            f"{len(samples['baseline'])} | {bench['replays_per_sample']} |")
    lines.extend(["", f"[Raw evidence with all timing samples and case results]({quote(raw_filename)})", "",
        "[Reproduction instructions](REPRODUCE.md) · [Recorded environment](environment.json) · "
        "[Candidate patch](candidate.patch)", "",
        "## Numerical checks and limits", "",
        "The harness compares the candidate with the original implementation and an FP64 reference. "
        "Its unit-test tolerances (rtol, atol) are:", ""])
    for dtype, values in sorted(clean["tolerances_rtol_atol"].items()):
        lines.append(f"- {dtype}: ({values[0]:.8g}, {values[1]:.8g})")
    lines.extend(["", "These tolerances are not the competition's acceptance criteria. "
        "Results on an RTX 5060, or any other local GPU named above, do not establish H200 "
        "performance, full-model numerical correctness, TP8 behavior, or end-to-end serving speed. "
        "This demonstration neither uses the official evaluator nor constitutes a competition submission. "
        "Timing variability is shown rather than treated as a confidence interval.", ""])
    return "\n".join(lines), clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Markdown file to write")
    parser.add_argument("--public-json", type=Path, required=True, help="Path-free evidence file to write")
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in (args.result, args.output, args.public_json)]
        require(len(set(paths)) == 3, "Input and output files must be distinct")
        require(paths[1].parent == paths[2].parent, "Keep both output files in the same directory")
        report = json.loads(args.result.read_text(encoding="utf-8-sig"))
        markdown, clean = summarize(report, args.public_json.name)
        # Validation finishes before either publishable file is written.
        args.public_json.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        args.output.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        print("No validated demonstration generated: " + str(exc), file=sys.stderr)
        return 2
    print("Validated operator demonstration and path-free evidence written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
