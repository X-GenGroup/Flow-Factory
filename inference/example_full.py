# Modified from https://huggingface.co/black-forest-labs/FLUX.1-dev usage example
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

# 1. Load pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

# 2. Load full fine-tuned transformer weights
checkpoint = 'path/to/checkpoint'  # replace with your checkpoint directory
pipe.transformer = FluxTransformer2DModel.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()

# 3. Generate image
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev-full.png")