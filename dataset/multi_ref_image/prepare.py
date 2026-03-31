# Build a multi-reference-image dataset from `sharegpt4o_image_mini`.
# Automatically downloads the source dataset if needed, then generates
# random 2-3 image combinations and symlinks the images directory.
#
# Usage:
#   python prepare.py
import json
import os
import random
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "..", "sharegpt4o_image_mini")
SOURCE_IMAGES_DIR = os.path.join(SOURCE_DIR, "images")
LOCAL_IMAGES_LINK = os.path.join(SCRIPT_DIR, "images")

if not os.path.isdir(SOURCE_IMAGES_DIR):
    print(f"Images not found at {SOURCE_IMAGES_DIR}, running download.sh ...")
    download_script = os.path.join(SOURCE_DIR, "download.sh")
    subprocess.run(["bash", download_script], check=True)
    assert os.path.isdir(SOURCE_IMAGES_DIR), (
        f"download.sh finished but {SOURCE_IMAGES_DIR} still missing"
    )
else:
    print(f"Images already present at {SOURCE_IMAGES_DIR}, skipping download.")

if not os.path.exists(LOCAL_IMAGES_LINK):
    os.symlink(os.path.relpath(SOURCE_IMAGES_DIR, SCRIPT_DIR), LOCAL_IMAGES_LINK)
    print(f"Created symlink {LOCAL_IMAGES_LINK} -> {SOURCE_IMAGES_DIR}")

trivia_prompt = "Combine these images together."
for split in ["train", "test"]:
    src_file = os.path.join(SOURCE_DIR, f"{split}.jsonl")
    dst_file = os.path.join(SCRIPT_DIR, f"{split}.jsonl")

    with open(src_file, "r") as f:
        data = [json.loads(line) for line in f]

    all_images = [item["image"] for item in data]
    seen_combinations = set()
    new_data = []
    i = 0
    data_num = len(data)
    while len(new_data) < data_num:
        random.seed(42 + i)
        ref_image_num = random.randint(2, 3)
        ref_images = random.sample(all_images, ref_image_num)

        signature = tuple(sorted(ref_images))

        if signature not in seen_combinations:
            new_data.append({"prompt": trivia_prompt, "images": ref_images})
            seen_combinations.add(signature)

        i += 1

    with open(dst_file, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(new_data)} entries to {dst_file}")