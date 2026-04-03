import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from torch.testing._internal.optests import opcheck
import unittest

from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn


def reference_muladd(a, b, c):
    return a * b + c


def get_extension(ext_name):
    if ext_name == "extension_cpp":
        import extension_cpp
        return extension_cpp
    else:
        import extension_cpp_stable
        return extension_cpp_stable


class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device, ext):
        samples = self.sample_inputs(device)
        for args in samples:
            result = ext.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    def test_correctness_cpu(self, ext_name):
        self._test_correctness("cpu", get_extension(ext_name))

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self, ext_name):
        self._test_correctness("cuda", get_extension(ext_name))

    def _test_gradients(self, device, ext):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = ext.ops.mymuladd(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_muladd(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)

            torch.testing.assert_close(result, expected)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    def test_gradients_cpu(self, ext_name):
        self._test_gradients("cpu", get_extension(ext_name))

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self, ext_name):
        self._test_gradients("cuda", get_extension(ext_name))

    def _opcheck(self, device, ext_name):
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        op = getattr(torch.ops, ext_name).mymuladd.default
        for args in samples:
            opcheck(op, args)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    def test_opcheck_cpu(self, ext_name):
        self._opcheck("cpu", ext_name)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self, ext_name):
        self._opcheck("cuda", ext_name)


class TestMyAddOut(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), make_tensor(3)],
            [make_tensor(20), make_tensor(20), make_tensor(20)],
        ]

    def _test_correctness(self, device, ext):
        samples = self.sample_inputs(device)
        for args in samples:
            result = args[-1]
            ext.ops.myadd_out(*args)
            expected = torch.add(*args[:2])
            torch.testing.assert_close(result, expected)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    def test_correctness_cpu(self, ext_name):
        self._test_correctness("cpu", get_extension(ext_name))

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self, ext_name):
        self._test_correctness("cuda", get_extension(ext_name))

    def _opcheck(self, device, ext_name):
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        op = getattr(torch.ops, ext_name).myadd_out.default
        for args in samples:
            opcheck(op, args)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    def test_opcheck_cpu(self, ext_name):
        self._opcheck("cpu", ext_name)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self, ext_name):
        self._opcheck("cuda", ext_name)


class TestTorchCompileStreamSync(TestCase):
    """Test for GitHub issue pytorch/pytorch#157363 - stream synchronization with torch.compile"""

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_compile_with_linear_layer(self, ext_name):
        """Test custom CUDA kernels with nn.Linear + torch.compile (the original failing case)"""
        ext = get_extension(ext_name)

        class Model(nn.Module):
            def __init__(self, size, extension):
                super().__init__()
                self.linear = nn.Linear(
                    size, size, device="cuda", dtype=torch.float32
                )
                self.ext = extension

            def forward(self, x):
                return self.ext.ops.mymuladd(self.linear(x), self.linear(x), 0.0)

        # Test sizes that previously failed
        for size in [1000, 5000, 10000]:
            with self.subTest(size=size):
                torch.manual_seed(42)
                model = Model(size, ext)
                x = torch.randn((1, size), device="cuda", dtype=torch.float32)

                with torch.no_grad():
                    expected = model(x)
                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    actual = compiled_model(x)

                self.assertEqual(actual, expected)

    @parametrize("ext_name", ["extension_cpp", "extension_cpp_stable"])
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_compile_custom_only(self, ext_name):
        """Test custom operations alone with torch.compile"""
        ext = get_extension(ext_name)

        def model(x):
            return ext.ops.mymuladd(x, x, 1.0)

        for size in [1000, 5000, 10000]:
            with self.subTest(size=size):
                torch.manual_seed(42)
                x = torch.randn((size,), device="cuda", dtype=torch.float32)

                with torch.no_grad():
                    expected = model(x)
                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    actual = compiled_model(x)

                self.assertEqual(actual, expected)


instantiate_parametrized_tests(TestMyMulAdd)
instantiate_parametrized_tests(TestMyAddOut)
instantiate_parametrized_tests(TestTorchCompileStreamSync)


if __name__ == "__main__":
    run_tests()
