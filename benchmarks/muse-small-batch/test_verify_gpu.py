"""Unit checks for dispatch observation; no Torch, Triton or GPU required."""
import unittest

from verify_gpu import KERNEL_NAME, checked_kernel_call


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(value, *, scale):
            self.calls.append((grid, value, scale))
            return value * scale
        return launch


def wrapper(kernel):
    namespace = {KERNEL_NAME: kernel}
    exec(
        "def invoke(value, fused=True, fail=False):\n"
        "    if fail:\n"
        "        raise RuntimeError('deliberate failure')\n"
        "    if fused:\n"
        f"        return {KERNEL_NAME}[(7,)](value, scale=3)\n"
        "    return value\n",
        namespace,
    )
    return namespace["invoke"]


class DispatchChecks(unittest.TestCase):
    def setUp(self):
        self.kernel = FakeKernel()
        self.fn = wrapper(self.kernel)

    def test_launch_forwards_grid_arguments_and_result(self):
        result, count = checked_kernel_call(self.fn, {"value": 4}, 1)
        self.assertEqual((result, count), (12, 1))
        self.assertEqual(self.kernel.calls, [((7,), 4, 3)])
        self.assertIs(self.fn.__globals__[KERNEL_NAME], self.kernel)

    def test_fallback_does_not_launch(self):
        result, count = checked_kernel_call(self.fn, {"value": 4, "fused": False}, 0)
        self.assertEqual((result, count), (4, 0))
        self.assertEqual(self.kernel.calls, [])

    def test_unchanged_fallback_cannot_pass_fused_route_check(self):
        with self.assertRaisesRegex(AssertionError, "Expected 1 kernel launches, observed 0"):
            checked_kernel_call(self.fn, {"value": 4, "fused": False}, 1)
        self.assertIs(self.fn.__globals__[KERNEL_NAME], self.kernel)

    def test_unexpected_launch_cannot_pass_fallback_check(self):
        with self.assertRaisesRegex(AssertionError, "Expected 0 kernel launches, observed 1"):
            checked_kernel_call(self.fn, {"value": 4}, 0)
        self.assertIs(self.fn.__globals__[KERNEL_NAME], self.kernel)

    def test_failure_restores_real_kernel(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            checked_kernel_call(self.fn, {"value": 4, "fail": True}, 0)
        self.assertIs(self.fn.__globals__[KERNEL_NAME], self.kernel)


if __name__ == "__main__":
    unittest.main()
