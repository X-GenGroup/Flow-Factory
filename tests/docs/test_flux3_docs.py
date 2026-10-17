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

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_lists_flux3_klein_models() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for repo in (
        "https://huggingface.co/black-forest-labs/flux-3-image-klein-base",
        "https://huggingface.co/black-forest-labs/flux-3-image-klein",
    ):
        row_start = readme.index(repo)
        model_row = readme[row_start : readme.index("</tr>", row_start)]
        assert "<td>5.5B</td>" in model_row
        assert "<td>flux3</td>" in model_row
