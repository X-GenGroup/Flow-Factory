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


def test_readme_lists_qwen_image_21_as_7b() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_link = '<a href="https://huggingface.co/Qwen/Qwen-Image-2.1">Qwen-Image 2.1</a>'
    row_start = readme.index(model_link)
    model_row = readme[row_start : readme.index("</tr>", row_start)]

    assert "<td>7B</td>" in model_row
    assert "<td>20B</td>" not in model_row
    assert "<td>qwen-image-2.1</td>" in model_row
