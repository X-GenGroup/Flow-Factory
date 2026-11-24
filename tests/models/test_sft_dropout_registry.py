"""Every built-in offline adapter identifies its SFT numeric and native policies."""

import pytest

from flow_factory.models.registry import get_model_adapter_class, list_registered_models
from flow_factory.models.sft_dropout import resolve_sft_dropout


@pytest.mark.parametrize("model_id", sorted(list_registered_models()))
def test_registered_sft_model_declares_admissible_dropout(model_id):
    if model_id == "bagel":
        pytest.importorskip(
            "flash_attn", reason="BAGEL requires its optional FlashAttention runtime"
        )
    cls = get_model_adapter_class(model_id)
    policy = resolve_sft_dropout(cls, "other/fixture", 0.1)
    assert policy.revision
    assert policy.supports_native == (model_id in {"bagel", "sensenova"})
    assert resolve_sft_dropout(cls, "other/fixture", 0) is None
    with pytest.raises(ValueError, match="checkpoint|qualified|native"):
        resolve_sft_dropout(cls, "other/fixture", "native")
