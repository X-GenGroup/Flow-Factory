# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import pytest
import torch

from flow_factory.models.bagel._parallel import PackedLayout


@pytest.mark.parametrize("lengths", [(6, 10), (0, 9, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize("cp,sp", [(1, 2), (2, 1), (2, 2)])
def test_packed_padding_preserves_samples_and_exact_backward(lengths, cp, sp):
    layout = PackedLayout(lengths, cp, sp)
    source = torch.arange(sum(lengths) * 3, dtype=torch.float32).reshape(-1, 3).requires_grad_()
    padded = layout.reorder(source)
    restored = layout.restore(padded)
    torch.testing.assert_close(restored, source, rtol=0, atol=0)
    restored.square().sum().backward()
    torch.testing.assert_close(source.grad, 2 * source, rtol=0, atol=0)
    for rank in range(cp):
        ids = layout.ids(rank, device="cpu")
        assert ids.numel() == sum(layout.local_lengths)
        assert ids.numel() % sp == 0
        offset = 0
        for length, local, part in zip(
            lengths, layout.local_lengths, ids.split(layout.local_lengths)
        ):
            assert ((part == -1) | ((part >= offset) & (part < offset + length))).all()
            offset += length


def test_packed_roles_keep_empty_reference_round_samples_and_padding_separate():
    layout = PackedLayout((0, 7, 5), 2, 2)
    text_indexes = torch.tensor([0, 6, 7, 11])
    for rank in range(2):
        ids = layout.ids(rank, device="cpu")
        text, gen = layout.roles(text_indexes, rank)
        assert set(text.tolist()).isdisjoint(gen.tolist())
        assert sorted(text.tolist() + gen.tolist()) == list(range(ids.numel()))
        assert all(int(ids[index]) in text_indexes.tolist() for index in text)
        assert all(int(ids[index]) not in text_indexes.tolist() for index in gen)


def test_cached_key_mask_follows_each_sample_and_native_cp_rank_order():
    layout = PackedLayout((6, 10), 2, 2)
    mask = layout.sample_key_mask(1, 5, device="cpu").flatten().tolist()
    # Prefix5 pads to6; target10 pads to12. Each rank contributes3+6 keys.
    assert mask == [True] * 9 + [True, True, False, True, True, True, True, False, False]
    assert sum(mask) == 15


@pytest.mark.parametrize(
    "lengths,cp,sp", [((-1,), 2, 1), ((True,), 2, 1), ((4,), 0, 1), ((4,), 2, 0)]
)
def test_invalid_packed_geometry_fails_before_mutation(lengths, cp, sp):
    with pytest.raises(ValueError):
        PackedLayout(lengths, cp, sp)
