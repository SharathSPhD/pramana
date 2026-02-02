from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "spaces"
    / "pramana-nyaya-demo"
    / "generation_params.py"
)
spec = spec_from_file_location("generation_params", MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
import sys

sys.modules[spec.name] = module
spec.loader.exec_module(module)
normalize_generation_params = module.normalize_generation_params


def test_normalize_generation_params_clamps_values():
    params = normalize_generation_params(
        max_new_tokens=5000,
        temperature=-1.0,
        top_p=1.5,
        top_k=999,
    )

    assert params.max_new_tokens == 1024
    assert params.temperature == 0.0
    assert params.top_p == 1.0
    assert params.top_k == 200
    assert params.do_sample is False


def test_normalize_generation_params_enables_sampling():
    params = normalize_generation_params(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    assert params.do_sample is True
