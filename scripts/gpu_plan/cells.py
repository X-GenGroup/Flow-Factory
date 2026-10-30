"""Cells for the PR #267 GPU plan.

Geometry for the distributed image cells is the plan's DDP layout:
``global_batch``, world size 2, group size 2, per-device batch size 1,
SD3.5, 4 denoising steps, ``num_sde_steps=2``. One probe launch rolls the
first microbatch (two, for online DPO) and then the algorithm's noise path.
It does not run a full epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

SD35 = "stabilityai/stable-diffusion-3.5-medium"
H3_MODEL = "MiniMaxAI/MiniMax-H3"
PROMPTS = "dataset/pickscore"
SFT_DATA = "dataset/sft_sd3_5"
OFFLINE_DPO_DATA = "dataset/offline_dpo_sd3_5"
H3_PROMPTS = "dataset/vid_prompt"
DUMMY_REWARD = "flow_factory.rewards.my_reward.MyPointwiseRewardModel"

SECTION_REPRO = "reproducibility"
SECTION_DIVERSITY = "diversity"
SECTION_ISOLATION = "isolation"


@dataclass(frozen=True)
class Seeds:
    """The three seeds the plan names. A cell changes at most one of them."""

    train: int = 42
    scheduler: int = 7
    eval: int = 11


@dataclass(frozen=True)
class Cell:
    """One named GPU-plan cell."""

    name: str
    kind: str
    world_size: int
    checklist: str
    sections: tuple[str, ...]
    captures: tuple[str, ...]
    init_same_noise: bool = False
    use_shared_noise: bool | None = None
    microbatches: int = 1
    accelerate_config: str | None = None


def _cell(name: str, kind: str, world: int, checklist: str, captures: tuple[str, ...], **kwargs):
    sections = kwargs.pop("sections", (SECTION_REPRO, SECTION_DIVERSITY))
    return Cell(
        name=name,
        kind=kind,
        world_size=world,
        checklist=checklist,
        sections=sections,
        captures=captures,
        **kwargs,
    )


CELLS: tuple[Cell, ...] = (
    _cell(
        "grpo_defaults",
        "grpo",
        2,
        "GRPO defaults",
        ("rollout", "plan", "eval"),
    ),
    _cell(
        "grpo_explicit",
        "grpo",
        2,
        "GRPO explicit",
        ("rollout", "plan"),
        init_same_noise=True,
    ),
    _cell(
        "grpo_rows",
        "grpo_rows",
        1,
        "GRPO explicit row swap",
        ("row_swap",),
        init_same_noise=True,
    ),
    _cell(
        "dgpo",
        "dgpo",
        2,
        "DGPO default",
        ("rollout", "plan", "dgpo"),
        use_shared_noise=True,
    ),
    _cell(
        "nft_off",
        "nft",
        2,
        "NFT flag off",
        ("rollout", "plan", "nft"),
        use_shared_noise=False,
    ),
    _cell(
        "nft_shared",
        "nft",
        2,
        "NFT shared noise",
        ("rollout", "plan", "nft"),
        use_shared_noise=True,
    ),
    _cell(
        "h3",
        "h3",
        8,
        "H3 request generator",
        ("h3",),
        accelerate_config="config/accelerate_configs/fsdp2.yaml",
    ),
    _cell(
        "sft",
        "sft",
        1,
        "SFT",
        ("sft",),
    ),
    _cell(
        "offline_dpo",
        "offline_dpo",
        1,
        "Offline DPO",
        ("offline_dpo",),
    ),
    _cell(
        "online_dpo",
        "dpo",
        2,
        "Online DPO",
        ("online_dpo",),
        microbatches=2,
    ),
    _cell(
        "crd",
        "crd",
        2,
        "CRD",
        ("rollout", "crd"),
    ),
    _cell(
        "tdm",
        "tdm",
        1,
        "TDM",
        ("tdm",),
    ),
    _cell(
        "tdm_h3",
        "tdm_h3",
        8,
        "TDM video and audio",
        ("tdm",),
        accelerate_config="config/accelerate_configs/fsdp2.yaml",
    ),
)

CELLS_BY_NAME = {cell.name: cell for cell in CELLS}


def require_cell(name: str) -> Cell:
    """Return one cell or raise with the known names."""

    cell = CELLS_BY_NAME.get(name)
    if cell is None:
        known = ", ".join(CELLS_BY_NAME)
        raise KeyError(f"unknown gpu-plan cell {name!r}; known cells: {known}")
    return cell


def _optimizer(name: str, *, update_frequency: int | None = None) -> dict:
    entry = {
        "name": name,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "max_grad_norm": 1.0,
    }
    if update_frequency is not None:
        entry["update_frequency"] = update_frequency
    return entry


def _rewards() -> list[dict]:
    return [
        {
            "name": "constant",
            "reward_model": DUMMY_REWARD,
            "batch_size": 1,
            "device": "cpu",
            "dtype": "float32",
        }
    ]


def render_config(cell: Cell, seeds: Seeds, run_name: str) -> dict:
    """Build the YAML document for one cell.

    ``gradient_accumulation_steps`` is 1 because the probe runs one
    microbatch and then stops. That is the opposite of a real experiment
    recipe, where ``auto`` is required so the epoch has one optimizer step.
    """

    if cell.kind in {"sft", "offline_dpo"}:
        data_dir = SFT_DATA if cell.kind == "sft" else OFFLINE_DPO_DATA
        dataset = {
            "name": "offline",
            "dataset_dir": data_dir,
            "train": {"weight": 1, "max_dataset_size": 2},
        }
        sampler = "auto"
        unique = 2
        group_size = 1
        per_device = 1
    elif cell.kind == "tdm":
        dataset = {
            "name": "default",
            "dataset_dir": PROMPTS,
            "train": {"weight": 1, "max_dataset_size": 8},
        }
        sampler = "auto"
        unique = 2
        group_size = 1
        per_device = 2
    elif cell.kind in {"h3", "tdm_h3"}:
        dataset = {
            "name": "vid_prompt",
            "dataset_dir": H3_PROMPTS,
            "train": {"weight": 1, "max_dataset_size": 8},
        }
        sampler = "auto"
        unique = 8
        group_size = 1
        per_device = 1
    elif cell.kind == "grpo_rows":
        dataset = {
            "name": "default",
            "dataset_dir": PROMPTS,
            "train": {"weight": 1, "max_dataset_size": 8},
        }
        sampler = "global_batch"
        unique = 1
        group_size = 2
        per_device = 2
    else:
        dataset = {
            "name": "default",
            "dataset_dir": PROMPTS,
            "train": {"weight": 1, "max_dataset_size": 8},
        }
        if "eval" in cell.captures:
            dataset["eval"] = {"max_dataset_size": 1}
        sampler = "global_batch"
        unique = 8
        group_size = 2
        per_device = 1

    online = cell.kind in {"grpo", "grpo_rows", "dgpo", "nft", "dpo", "crd", "h3"}
    image = cell.kind not in {"h3", "tdm_h3"}
    model: dict = {
        "finetune_type": "lora",
        "target_modules": "default",
        "resume_path": None,
        "resume_type": None,
    }
    if image:
        model.update(
            {
                "model_name_or_path": SD35,
                "model_type": "sd3-5",
                "lora_rank": 4,
                "lora_alpha": 8,
            }
        )
    else:
        model.update(
            {
                "model_name_or_path": H3_MODEL,
                "model_type": "minimax-h3-t2va",
                "component_load_dtypes": "bf16",
                "lora_rank": 16,
                "lora_alpha": 32,
                "target_components": "transformer",
                "trainable_parameters_dtype": "bf16",
            }
        )

    train: dict = {
        "trainer_type": _trainer_type(cell),
        "max_epochs": 1,
        "per_device_batch_size": per_device,
        "group_size": group_size,
        "unique_sample_num_per_epoch": unique,
        "gradient_step_per_epoch": 1,
        "gradient_accumulation_steps": 1,
        "num_inner_epochs": 1,
        "shuffle_samples": False,
        "ema_decay": 0.0,
        "ema_update_interval": 1,
        "ema_device": "cpu",
        "enable_gradient_checkpointing": cell.kind in {"h3", "tdm_h3", "dpo"},
        "seed": seeds.train,
    }
    if image:
        train["resolution"] = 512
        train["guidance_scale"] = 1.0 if cell.kind in {"sft", "offline_dpo", "tdm"} else 4.5
        train["num_inference_steps"] = 2 if cell.kind == "tdm" else 4
    else:
        # 124 frames is the shortest legal clip: 5 seconds at a fixed 24 fps,
        # snapped to the video VAE rule 17*n+5. 64x96 is the validated debug canvas.
        train["resolution"] = [64, 96]
        train["num_frames"] = 124
        train["frame_rate"] = 24.0
        train["guidance_scale"] = 1.0
        train["num_inference_steps"] = 2
        train["offload_samples_to_cpu"] = True
    if cell.kind in {"grpo", "grpo_rows", "h3"}:
        train["init_same_noise"] = cell.init_same_noise
        train["kl_beta"] = 0
        train["ref_param_device"] = "cpu"
        train["advantage_aggregation"] = "gdpo"
    if cell.kind == "dgpo":
        train["use_shared_noise"] = True if cell.use_shared_noise is None else cell.use_shared_noise
        train["kl_beta"] = 0
        train["ref_param_device"] = "cpu"
        train["dpo_beta"] = 100.0
    if cell.kind == "nft":
        train["use_shared_noise"] = bool(cell.use_shared_noise)
        train["num_train_timesteps"] = 2
        train["time_sampling_strategy"] = "logit_normal"
        train["kl_beta"] = 0
        train["ref_param_device"] = "cpu"
    if cell.kind == "dpo":
        train["num_train_timesteps"] = 2
        train["beta"] = 100.0
        train["timestep_range"] = 0.99
        train["ref_param_device"] = "cpu"
    if cell.kind == "crd":
        train["num_train_timesteps"] = 2
        train["time_sampling_strategy"] = "logit_normal"
        train["kl_beta"] = 0
        train["ref_param_device"] = "cpu"
    if cell.kind in {"sft", "offline_dpo"}:
        train["num_train_timesteps"] = 2
        train["weighting_scheme"] = "logit_normal"
        train["timestep_range"] = 0.99
        train["time_shift"] = 1.0
        train["guidance_scale"] = 1.0
    if cell.kind == "offline_dpo":
        train["beta"] = 2000.0
        train["ref_param_device"] = "cpu"
    if cell.kind in {"tdm", "tdm_h3"}:
        train.update(
            {
                "tdm_query_interval": "trajectory",
                "tdm_query_max_sigma": 1.0,
                "tdm_query_distribution": "conditional_logit_normal",
                "tdm_query_logit_mean": 0.0,
                "tdm_query_logit_std": 1.0,
                "real_guidance_scale": 1.0 if cell.kind == "tdm_h3" else 4.0,
                "use_huber": True,
                "huber_c": 0.001,
                "tdm_snr_gamma": 5.0,
                "tdm_importance_clip": 20.0,
            }
        )
        train["gradient_accumulation_steps"] = int(train["num_inference_steps"])

    if cell.kind in {"tdm", "tdm_h3"}:
        scheduler = {"dynamics_type": "ODE", "seed": seeds.scheduler}
    elif cell.kind in {"sft", "offline_dpo"}:
        scheduler = {"dynamics_type": "ODE", "seed": seeds.scheduler}
    elif cell.kind == "h3":
        scheduler = {
            "dynamics_type": "CPS",
            "noise_level": 0.8,
            "num_sde_steps": 2,
            "sde_steps": [0, 1],
            "seed": seeds.scheduler,
        }
    else:
        scheduler = {
            "dynamics_type": "Flow-SDE",
            "noise_level": 0.8,
            "num_sde_steps": 2,
            "sde_steps": [0, 1, 2, 3],
            "seed": seeds.scheduler,
        }

    document = {
        "launcher": "accelerate",
        "num_processes": cell.world_size,
        "main_process_port": 29621,
        "mixed_precision": "bf16",
        "data": {
            "datasets": [dataset],
            "enable_preprocess": True,
            "preprocessing_batch_size": 1,
            "dataloader_num_workers": 0,
            "force_reprocess": False,
            "cache_dir": "~/.cache/flow_factory/datasets",
            "sampler_type": sampler,
        },
        "model": model,
        "log": {
            "run_name": run_name,
            "project": "Flow-Factory",
            "logging_backend": "none",
            "save_dir": "/tmp/ff-gpu-plan",
            "save_freq": 0,
            "save_model_only": True,
        },
        "train": train,
        "scheduler": scheduler,
        "eval": {
            "eval_freq": 0,
            "seed": seeds.eval,
            "resolution": train["resolution"],
            "num_inference_steps": train["num_inference_steps"],
            "guidance_scale": train["guidance_scale"],
            "per_device_batch_size": 1,
        },
        "optimizers": (
            [_optimizer("generator", update_frequency=1), _optimizer("fake", update_frequency=1)]
            if cell.kind in {"tdm", "tdm_h3"}
            else [_optimizer("default")]
        ),
    }
    if online:
        document["rewards"] = _rewards()
    if cell.kind in {"h3", "tdm_h3"}:
        document["eval"]["num_frames"] = 124
        document["eval"]["frame_rate"] = 24.0
    return document


def _trainer_type(cell: Cell) -> str:
    return {
        "grpo": "grpo",
        "grpo_rows": "grpo",
        "dgpo": "dgpo",
        "nft": "nft",
        "h3": "grpo",
        "sft": "sft",
        "offline_dpo": "offline-dpo",
        "dpo": "dpo",
        "crd": "crd",
        "tdm": "tdm",
        "tdm_h3": "tdm",
    }[cell.kind]
