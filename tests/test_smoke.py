"""Smoke test: the package imports. Real tests arrive per week.

Week 1 adds numeric-equivalence tests vs the HF reference (RMSNorm, RoPE, one
block). Week 2 adds the token-for-token greedy match. Week 3 adds sampler tests.
"""

import nanoserve


def test_version():
    assert nanoserve.__version__ == "0.0.0"
