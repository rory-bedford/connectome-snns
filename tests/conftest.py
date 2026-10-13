"""Shared pytest configuration.

The suite compares the simulators against fixtures recorded before any optimisation
(see ``tests/fixtures/generate.py``). Tests that need a GPU are marked ``gpu`` and
skipped without one; CPU tests use a single thread, because CPU matmul results can
depend on the thread count.
"""

import pytest
import torch

torch.set_num_threads(1)

HAS_CUDA = torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip)


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu),
]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param
