"""Synthetic fixtures test the report gate; these are not GPU measurements."""
import copy
import unittest

from summarize_demo import summarize


def fixture():
    return {
        "status": "PASS_GPU", "gpu_status": "PASS_GPU", "passed": 195, "failed": 0,
        "load_mode": "isolated_full_module", "baseline_ref": "a" * 40,
        "baseline_sha256": "b" * 64, "candidate_sha256": "c" * 64,
        "candidate_path": r"C:\Users\private-user\candidate.py",
        "torch": "2.9.0+cu128", "triton": "3.5.0", "cuda_runtime": "12.8", "hip_runtime": None,
        "device": {"name": "Synthetic GPU", "capability": [12, 0], "memory_bytes": 8 * 2**30},
        "results": [{"case": f"cuda/synthetic/{i}", "status": "PASS"} for i in range(195)],
        "tolerances_rtol_atol": {"float16": [0.002, 0.002], "bfloat16": [0.016, 0.016],
                                 "float32": [3e-5, 3e-6]},
        "benchmarks": [{"shape": [n, 8, 4096], "dtype": "bfloat16",
                         "median_us": {"baseline": 6, "candidate": 3},
                         "samples_us": {"baseline": [5, 6, 7], "candidate": [2, 3, 4]},
                         "replays_per_sample": 100} for n in (1, 8)],
    }


class SummaryChecks(unittest.TestCase):
    def test_summary_preserves_every_sample_and_excludes_local_path(self):
        source = fixture()
        markdown, clean = summarize(source, "evidence.json")
        self.assertNotIn("private-user", markdown + str(clean))
        self.assertNotIn("candidate_path", clean)
        self.assertEqual(clean["benchmarks"], [dict(b, scope=clean["scope"]) for b in source["benchmarks"]])
        self.assertIn("1 × 8 × 4096", markdown)
        self.assertIn("8 × 8 × 4096", markdown)
        self.assertIn("2×", markdown)
        self.assertIn("](evidence.json)", markdown)
        self.assertIn("do not establish H200", markdown)

    def test_incomplete_or_failed_runs_cannot_be_reported_as_validated(self):
        for field, value in (("status", "FAIL"), ("gpu_status", "NOT_RUN_GPU"),
                             ("failed", 1), ("passed", 194), ("benchmarks", [])):
            with self.subTest(field=field):
                source = fixture()
                source[field] = value
                with self.assertRaises(ValueError):
                    summarize(source, "evidence.json")

    def test_cpu_cases_cannot_satisfy_gpu_count(self):
        source = fixture()
        source["results"] = [{"case": f"cpu/synthetic/{i}", "status": "PASS"} for i in range(195)]
        with self.assertRaisesRegex(ValueError, "CUDA case rows"):
            summarize(source, "evidence.json")

    def test_claimed_count_must_match_rows(self):
        source = fixture()
        source["results"].pop()
        with self.assertRaisesRegex(ValueError, "pass count"):
            summarize(source, "evidence.json")

    def test_bad_timings_or_median_cannot_establish_speed(self):
        for change in (lambda b: b["samples_us"]["candidate"].__setitem__(0, float("nan")),
                       lambda b: b["median_us"].__setitem__("candidate", 0),
                       lambda b: b["median_us"].__setitem__("baseline", 600)):
            source = fixture()
            change(source["benchmarks"][0])
            with self.assertRaises(ValueError):
                summarize(source, "evidence.json")

    def test_duplicate_benchmark_is_rejected(self):
        source = fixture()
        source["benchmarks"][1] = copy.deepcopy(source["benchmarks"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(source, "evidence.json")

    def test_slower_candidate_is_reported_without_speedup_claim(self):
        source = fixture()
        for bench in source["benchmarks"]:
            bench["samples_us"]["candidate"] = [11, 12, 13]
            bench["median_us"]["candidate"] = 12
        markdown, _ = summarize(source, "evidence.json")
        self.assertIn("0.5×", markdown)
        self.assertNotIn("speedup", markdown)

    def test_raw_link_cannot_expose_a_local_path(self):
        with self.assertRaisesRegex(ValueError, "filename"):
            summarize(fixture(), r"C:\Users\private-user\evidence.json")


if __name__ == "__main__":
    unittest.main()
