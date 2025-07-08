import os
import time
import pickle
import torch
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from experiments.robot.openvla_utils import get_action_head, get_processor
from experiments.robot.robot_utils import get_model
from PIL import Image
import numpy as np
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    base_vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_actions_chunk: int = -999
    num_open_loop_steps: int = -999    # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    mode: str = "mul"
    num_actions_per_token: int = -999
    action_head_name: str = "mlp"
    hidden_dim: int = 4096
    num_blocks: int = 2
    # fmt: on

cfg = GenerateConfig(
    pretrained_checkpoint="juyil/libero_object-b8-3rd_person_img-8act-mul",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=1,
    use_proprio=False,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=8,
    unnorm_key="libero_object_no_noops",
    mode="mul",
    num_actions_chunk=8,
    num_actions_per_token=8,
)

vla :OpenVLAForActionPrediction = get_model(cfg).to("cuda:0").eval()
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
processor = get_processor(cfg)
vla.predict_action = vla.mul_predict_action

image = Image.open("example.png").convert("RGB")
prompt = "In: What action should the robot take to pick the cup?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# with open("./sample_libero_spatial_observation.pkl", "rb") as file:
#     observation = pickle.load(file)
# img_np = np.array(image)
# observation = {
#     "full_image": img_np,
#     "task_description": "pick the cup",
# }

for _ in range(5):
    actions, _ = vla.predict_action(
        **inputs,
        unnorm_key=cfg.unnorm_key,
        do_sample=False,
        proprio=False,
        action_head=action_head,
        use_film=False,
        cfg=cfg,
    )


torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    actions, _ = vla.predict_action(
        **inputs,
        unnorm_key=cfg.unnorm_key,
        do_sample=False,
        proprio=False,
        action_head=action_head,
        use_film=False,
        cfg=cfg,
    )

torch.cuda.synchronize()
end = time.time()

total = end - start
avg = total / 100
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"total time for 100 runs: {total:.4f}s")
print(f"average time per run: {avg:.4f}s")
print(f"peak VRAM usage: {peak_vram:.2f} GB")
print(actions)
