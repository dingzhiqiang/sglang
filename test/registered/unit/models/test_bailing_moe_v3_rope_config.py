import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.bailing_moe_v3 import _resolve_rope_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBailingMoeV3RopeConfig(unittest.TestCase):
    def test_legacy_top_level_rope_theta(self):
        config = SimpleNamespace(rope_theta=6_000_000)

        self.assertEqual(
            _resolve_rope_config(config),
            (6_000_000, None),
        )

    def test_none_rope_parameters_uses_legacy_theta(self):
        config = SimpleNamespace(
            rope_parameters=None, rope_theta=6_000_000, rope_scaling=None
        )
        self.assertEqual(_resolve_rope_config(config), (6_000_000, None))

    def test_legacy_config_constructs_rotary_embedding(self):
        theta, scaling = _resolve_rope_config(SimpleNamespace(rope_theta=6_000_000))
        with torch.device("cpu"), patch(
            "sglang.srt.layers.rotary_embedding.base.get_server_args",
            return_value=SimpleNamespace(rl_on_policy_target=None),
        ):
            rope = get_rope(
                head_size=64,
                rotary_dim=64,
                max_position=256,
                base=theta,
                rope_scaling=scaling,
                dtype=torch.float32,
            )

        self.assertEqual(rope.base, 6_000_000)
        self.assertEqual(rope.cos_sin_cache.shape, (256, 64))
        self.assertTrue(torch.isfinite(rope.cos_sin_cache).all())

    def test_preserves_bailing_default_theta(self):
        self.assertEqual(_resolve_rope_config(SimpleNamespace()), (600000, None))
        rope_parameters = {"rope_type": "default"}
        config = SimpleNamespace(rope_parameters=rope_parameters, rope_theta=6_000_000)
        self.assertEqual(_resolve_rope_config(config), (600000, rope_parameters))

    def test_malformed_rope_parameters_does_not_fall_back(self):
        for malformed in (False, [], "default"):
            with self.subTest(rope_parameters=malformed), self.assertRaises(
                AttributeError
            ):
                _resolve_rope_config(
                    SimpleNamespace(rope_parameters=malformed, rope_theta=6_000_000)
                )

    def test_rope_scaling_fallback_preserves_fields(self):
        rope_scaling = {"rope_type": "yarn", "factor": 4.0}
        config = SimpleNamespace(
            rope_scaling=rope_scaling,
            rope_theta=6_000_000,
        )

        rope_theta, normalized = _resolve_rope_config(config)

        self.assertEqual(rope_theta, 6_000_000)
        self.assertEqual(
            normalized,
            {"rope_type": "yarn", "factor": 4.0},
        )
        self.assertEqual(rope_scaling, {"rope_type": "yarn", "factor": 4.0})

    def test_rope_parameters_take_precedence_without_mutation(self):
        rope_parameters = {"rope_type": "default", "rope_theta": 1_000_000}
        config = SimpleNamespace(
            rope_parameters=rope_parameters,
            rope_scaling={"factor": 8.0},
            rope_theta=6_000_000,
        )

        rope_theta, normalized = _resolve_rope_config(config)

        self.assertEqual(rope_theta, 1_000_000)
        self.assertEqual(normalized, rope_parameters)
        self.assertIs(normalized, rope_parameters)


if __name__ == "__main__":
    unittest.main()
