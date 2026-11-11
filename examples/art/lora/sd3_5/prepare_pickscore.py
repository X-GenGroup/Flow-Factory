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

"""Prepare the pinned public ART PickScore recipe assets from the repository root."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

SNAPSHOTS = {
    "sd35": ("stabilityai/stable-diffusion-3.5-medium", "b940f670f0eda2d07fbb75229e779da1ad11eb80"),
}
PROMPTS = {
    "train": (
        "84b2fc9bb85c-train.txt",
        "2f425adcbb9a6d0feee7c6c448e1cb83a1f71deca9434b803907e2578eca837f",
    ),
    "test": (
        "f61ace68ff56-test.txt",
        "d75c025fa451866e70f7e36243f7cdd93095523cb51f5f9651f2ce3236d20262",
    ),
}


def main():
    assets = Path(".scratch/art/assets")
    assets.mkdir(parents=True, exist_ok=True)
    for name, (repo, revision) in SNAPSHOTS.items():
        snapshot = Path(snapshot_download(repo, revision=revision)).resolve()
        link = assets / name
        if link.exists():
            if link.resolve() != snapshot:
                raise ValueError(f"Existing asset {link} does not point to {snapshot}")
        else:
            link.symlink_to(snapshot, target_is_directory=True)
    data = Path("dataset/art_pickscore")
    data.mkdir(parents=True, exist_ok=True)
    for split, (filename, digest) in PROMPTS.items():
        url = (
            "https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/resolve/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/sources/"
            + filename
        )
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Prompt hash mismatch: {filename}")
        rows = [json.dumps({"prompt": line}) for line in raw.decode().splitlines() if line.strip()]
        (data / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
        print(f"{split}: {len(rows)} rows; source SHA256 {digest}")


if __name__ == "__main__":
    main()
