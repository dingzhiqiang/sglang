import unittest
from functools import wraps
from unittest.mock import patch

from sglang.jit_kernel.flash_attention_v3 import (
    _call_fa3_kernel,
    _is_default_optional_kwarg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashAttentionV3KernelCompatibility(unittest.TestCase):
    def test_drops_unsupported_default_kwargs(self):
        calls = []

        def old_kernel(value, supported=None):
            calls.append((value, supported))
            return value + 1

        result = _call_fa3_kernel(
            old_kernel,
            4,
            supported="kept",
            out=object(),
            only_qv=False,
        )

        self.assertEqual(result, 5)
        self.assertEqual(calls, [(4, "kept")])

    def test_rejects_dropping_required_semantics(self):
        def old_kernel(value):
            return value

        with self.assertRaisesRegex(TypeError, "required argument only_qv=True"):
            _call_fa3_kernel(old_kernel, 4, only_qv=True)

    def test_preserves_unrelated_type_errors(self):
        def broken_kernel(**kwargs):
            raise TypeError("invalid tensor layout")

        with self.assertRaisesRegex(TypeError, "invalid tensor layout"):
            _call_fa3_kernel(broken_kernel, optional_flag=False)

    def test_preserves_nested_unexpected_keyword_error(self):
        def inner_kernel():
            return None

        def wrapper(**kwargs):
            return inner_kernel(**kwargs)

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'out'"):
            _call_fa3_kernel(wrapper, out=object())

    def test_non_scalar_optional_value_is_not_treated_as_zero(self):
        class NonScalar:
            def __eq__(self, other):
                raise AssertionError("non-scalars must not be compared with zero")

        self.assertFalse(_is_default_optional_kwarg("only_qv", NonScalar()))

    def test_preserves_wrapped_kernel_errors(self):
        calls = []

        def inner_kernel():
            return None

        @wraps(inner_kernel)
        def wrapper(**kwargs):
            calls.append(kwargs)
            return inner_kernel(**kwargs)

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'out'"):
            _call_fa3_kernel(wrapper, out=object())
        self.assertEqual(len(calls), 1)

    def test_rejects_unknown_kwargs_even_when_false(self):
        def old_kernel():
            return None

        with self.assertRaisesRegex(TypeError, "required argument unknown=False"):
            _call_fa3_kernel(old_kernel, unknown=False)

    def test_rejects_uninspectable_kernel(self):
        def old_kernel():
            return None

        with patch(
            "sglang.jit_kernel.flash_attention_v3.inspect.signature",
            side_effect=ValueError("no signature"),
        ), self.assertRaisesRegex(TypeError, "unexpected keyword argument 'out'"):
            _call_fa3_kernel(old_kernel, out=object())


if __name__ == "__main__":
    unittest.main()
