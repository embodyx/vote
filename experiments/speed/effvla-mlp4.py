import os
import time
import pickle
import torch
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_vla_action
from experiments.robot.robot_utils import get_model
from PIL import Image
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cfg = GenerateConfig(
    pretrained_checkpoint="juyil/libero_10-b24-3rd_person_img-16act-mlp4-60ksteps",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=1,
    use_proprio=False,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=16,
    unnorm_key="libero_10_no_noops",
    mode="mul",
    num_actions_chunk=16,
    num_actions_per_token=16,
    action_head_name="mlp",
    hidden_dim=4096,
    num_blocks=4,
)

vla = get_model(cfg).to("cuda:0").eval()
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
processor = get_processor(cfg)
vla.predict_action = vla.mul_predict_action

image = Image.open("example.png").convert("RGB")
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# with open("./sample_libero_spatial_observation.pkl", "rb") as file:
#     observation = pickle.load(file)
img_np = np.array(image)
observation = {
    "full_image": img_np,
    "task_description": "pick the cup",
}

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
