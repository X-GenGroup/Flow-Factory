# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainers that optimize toward another model rather than toward a reward.

The target is a teacher, a frozen pretrained score, or an earlier copy of the
policy itself, so these algorithms have no reward or advantage stage; their
``prepare_feedback`` is a no-op. TDM-R1 is the exception: it adds reward-trained
surrogate terms to TDM and runs the reward and advantage stages. Reward-driven
algorithms live in ``trainers.rl``.

Modules are imported lazily through ``trainers.registry``, so nothing is
re-exported here; importing this package must not pull in every trainer.
"""
