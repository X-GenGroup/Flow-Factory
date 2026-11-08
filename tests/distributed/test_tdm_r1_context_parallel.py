# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Actual R1 trajectory, reward overlap and three-role CP continuation."""

from tdm_context_parallel_runner import qualify_lifecycle


def test_wan_tdm_r1_tp_sp_cp_overlap_roles_and_resume(tmp_path):
    groups = qualify_lifecycle(
        tmp_path,
        recipe_module="wan_tdm_r1_fixture",
        options={"algorithm": "tdm-r1", "dp_size": 2, "cp_size": 2, "tp_size": 2},
    )
    for leg in ("continuous", "resumed"):
        for cycle in range(1 if leg == "resumed" else 2):
            rows = [worker["records"]["overlap_cycles"][cycle] for worker in groups[leg]]
            assert sum(row["pending_before_fake"] for row in rows) > 0
            assert sum(row["tiles"][0]["pending"] for row in rows) > 0
            advantages = [
                value for row in rows for tile in row["tiles"] for value in tile["advantages"]
            ]
            assert min(advantages) < 0 < max(advantages)
            rewards = [value for row in rows for tile in row["tiles"] for value in tile["rewards"]]
            assert min(rewards) < max(rewards)


def test_reward_fixture_accepts_canonical_decoded_video_fields():
    import threading
    from types import SimpleNamespace

    import torch
    from wan_tdm_r1_fixture import GatedReward

    from flow_factory.hparams import RewardArguments
    from flow_factory.rewards import RewardProcessor
    from flow_factory.samples import BaseSample

    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1, process_index=0)
    args = RewardArguments(name="probe", device="cpu", batch_size=1)
    reward = GatedReward(args, accelerator)
    event = threading.Event()
    event.set()
    before = GatedReward.controller
    GatedReward.controller = {"planned": event, "ready": [event], "indices": {5: 0}}
    try:
        processor = RewardProcessor(accelerator, {"probe": reward}, {"probe": args}, verbose=False)
        sample = BaseSample(
            video=torch.ones(2, 3, 4, 4),
            sampling_sample_id=5,
            sampling_group_id=10,
            sampling_group_member_id=0,
        )
        actual = processor._gated_compute("probe", reward, [sample])
        torch.testing.assert_close(actual, torch.tensor([-1.0]))
    finally:
        GatedReward.controller = before
