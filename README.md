<h1 align="center">
  <img src="./assets/logo.png" alt="logo" height="100" style="vertical-align: middle;">
  Flow-Factory
</h1>

<!-- <p align="center">
  <img src="./assets/logo.png" alt="Flow-Factory Logo" width="200">
</p>

<h1 align="center"> Flow-Factory </h1> -->

<p align="center">
  <b>Easy Reinforcement Learning for Diffusion and Flow-Matching Models</b>
</p>

# Table of Contents

- [Supported Models](#-supported-models)
- [Supported Algorithms](#-supported-algorithms)
- [Get Started](#-get-started)
  - [Installation](#installation)
  - [Experiment Trackers](#experiment-trackers)
  - [Quick Start Example](#quick-start-example)
- [Dataset](#-dataset)
  - [Text-to-Image & Text-to-Video](#text-to-image--text-to-video)
  - [Image-to-Image & Image-to-Video](#image-to-image--image-to-video)
  - [Video-to-Video](#video-to-video)
- [Reward Model](#-reward-model)
- [Acknowledgements](#acknowledgements)

# 🤗 Supported Models

<table>
  <tr><th>Task</th><th>Model</th><th>Model Size</th><th>Model Type</th></tr>
  <tr><td rowspan="4">Text-to-Image</td><td><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">FLUX.1-dev</a></td><td>13B</td><td>flux1</td></tr>
  <tr><td><a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo">Z-Image-Turbo</a></td><td>12B</td><td>z-image</td></tr>
  <tr><td><a href="https://huggingface.co/Qwen/Qwen-Image">Qwen-Image</a></td><td>20B</td><td>qwen-image</td></tr>
  <tr><td><a href="https://huggingface.co/Qwen/Qwen-Image-2512">Qwen-Image-2512</a></td><td>20B</td><td>qwen-image</td></tr>
  <tr><td>Text-to-Image & Image(s)-to-Image</td><td><a href="https://huggingface.co/black-forest-labs/FLUX.2-dev">FLUX.2-dev</a></td><td>30B</td><td>flux2</td></tr>
  <tr><td>Image-to-Image</td><td><a href="https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev">FLUX.1-Kontext-dev</a></td><td>13B</td><td>flux1-kontext</td></tr>
  <tr><td rowspan="2">Image(s)-to-Image</td><td><a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2509">Qwen-Image-Edit-2509</a></td><td>20B</td><td>qwen-image-edit-plus</td></tr>
  <tr><td><a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2511">Qwen-Image-Edit-2511</a></td><td>20B</td><td>qwen-image-edit-plus</td></tr>
  <tr><td rowspan="3">Text-to-Video</td><td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">Wan2.1-T2V-1.3B</a></td><td>1.3B</td><td>wan2_t2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">Wan2.1-T2V-14B</a></td><td>14B</td><td>wan2_t2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers">Wan2.2-T2V-A14B</a></td><td>A14B</td><td>wan2_t2v</td></tr>
  <tr><td rowspan="4">Image-to-Video</td><td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers">Wan2.1-I2V-14B-480P</a></td><td>14B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers">Wan2.1-I2V-14B-480P</a></td><td>14B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">Wan2.1-I2V-14B-720P</a></td><td>14B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers">Wan2.2-I2V-A14B</a></td><td>A14B</td><td>wan2_i2v</td></tr>
</table>

# 💻 Supported Algorithms

| Algorithm      | `trainer_type` |
|----------------|----------------|
| GRPO           | grpo           |
| GRPO-Guard     | grpo-guard     |
| DiffusionNFT   | nft            |

See [`Algorithm Guidance`](guidance/algorithms.md) for more information.


# 💾 Hardward Requirements

# 🚀 Get Started

## Installation

```bash
git clone https://github.com/Jayce-Ping/Flow-Factory.git
cd Flow-Factory
pip install -e .
```

Optional dependencies, such as `deepspeed`, are also available. Install them with:

```bash
pip install -e .[deepspeed]
```

## Experiment Trackers

To use [Weights & Biases](https://wandb.ai/site/) or [SwanLab](https://github.com/SwanHubX/SwanLab) to log experimental results, install extra dependencies via `pip install -e .[wandb]` or `pip install -e .[swanlab]`.

> [SwanLab](https://github.com/SwanHubX/SwanLab) is recommended for users in mainland China.

After installation, set corresponding arguments in the config file:

```yaml
run_name: null  # Run name (auto: {model_type}_{finetune_type}_{timestamp})
project: "Flow-Factory"  # Project name for logging
logging_backend: "wandb"  # Options: wandb, swanlab, none
```

These trackers allow you to visualize both **training samples** and **metric curves** online:

![Online Image Samples](assets/wandb_images.png)

![Online Metric Examples](assets/wandb_metrics.png)

## Quick Start Example

Start training with the following simple command:

```bash
ff-train examples/grpo/lora/flux.yaml
```


# 📊 Dataset

The unified structure of dataset is:

```plaintext
|---- dataset
|----|--- train.txt / train.jsonl
|----|--- test.txt / test.jsonl (optional)
|----|--- images (optional)
|----|---| image1.png
|----|---| ...
|----|--- videos (optional)
|----|---| video1.mp4
|----|---| ...
```

## Text-to-Image & Text-to-Video

For text-to-image and text-to-video tasks, the only required input is the **prompt** in plain text format. Use `train.txt` and `test.txt` (optional) with following format:

```
A hill in a sunset.
An astronaut riding a horse on Mars.
```
> Example: [dataset/pickscore](./dataset/pickscore/train.txt)

Each line represents a single text prompt. Alternatively, you can use `train.jsonl` and `test.jsonl` in the following format:

```jsonl
{"prompt": "A hill in a sunset."}
{"prompt": "An astronaut riding a horse on Mars."}
```

> Example: [dataset/t2is](./dataset/t2is/train.jsonl)

## Image-to-Image & Image-to-Video

For tasks involving conditioning images, use `train.jsonl` and `test.jsonl` in the following format:

```jsonl
{"prompt": "A hill in a sunset.", "image": "path/to/image1.png"}
{"prompt": "An astronaut riding a horse on Mars.", "image": "path/to/image2/png"}
```

> Example: [dataset/sharegpt4o_image_mini](./dataset/sharegpt4o_image_mini/train.jsonl)

The default root directory for images is `dataset_dir/images`, and for videos, it is `dataset_dir/videos`. You can override these locations by setting the `image_dir` and `video_dir` variables in the config file:

```yaml
data:
    dataset_dir: "path/to/dataset"
    image_dir: "path/to/image_dir" # (default to "{dataset_dir}/images")
    video_dir: "path/to/video_dir" # (default to "{dataset_dir}/videos")
```

For models like [FLUX.2-dev]((https://huggingface.co/black-forest-labs/FLUX.2-dev)) and [Qwen-Image-Edit-2511]((https://huggingface.co/Qwen/Qwen-Image-Edit-2511)) that are able to accept multiple images as conditions, use the `images` key with a list of image paths:

```jsonl
{"prompt": "A hill in a sunset.", "images": ["path/to/condition_image_1_1.png", "path/to/condition_image_1_2.png"]}
{"prompt": "An astronaut riding a horse on Mars.", "images": ["path/to/condition_image_2_1.png", "path/to/condition_image_2_2.png"]}
```

## Video-to-Video

```jsonl
{"prompt": "A hill in a sunset.", "video": "path/to/video1.png"}
{"prompt": "An astronaut riding a horse on Mars.", "videos": ["path/to/video2.png", "path/to/video3.png"]}
```

# 💯 Reward Model

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Built-in Reward Models

The following reward models are pre-registered and ready to use:

| Name | Description | Reference |
|------|-------------|-----------|
| `PickScore` | CLIP-based aesthetic scoring model | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |

## Using Built-in Reward Models

Simply specify the reward model name in your config file:
```yaml
reward:
  reward_model: "PickScore"
  dtype: "bfloat16"
  device: "cuda"
  batch_size: 16
```

Refer to [Rewards Guidance](guidance/rewards.md) for more information about advanced usage, such as creating a custom reward model.


# Acknowledgements

This repository is based on [diffusers](https://github.com/huggingface/diffusers/), [accelerate](https://github.com/huggingface/accelerate) and [peft](https://github.com/huggingface/peft).
We thank them for their contributions to the community!!