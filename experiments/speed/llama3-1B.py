import os
import time
import torch
from experiments.robot.openvla_utils import get_action_head, get_processor,GenerateConfig
from experiments.robot.robot_utils import get_model
from PIL import Image
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


os.environ["TOKENIZERS_PARALLELISM"] = "false"



cfg = GenerateConfig(
    base_vla_path="juyil/llama3.2-1B-VLM",
    pretrained_checkpoint="juyil/llama3.2-1B-spatial",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=1,
    use_proprio=False,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=8,
    unnorm_key="libero_spatial_no_noops",
    mode="mul",
    num_actions_chunk=8,
    num_actions_per_token=8,
    action_head_name="fel",
    num_blocks=4,
    model_type="llama3.2",
    hidden_dim=2048,
)

vla :OpenVLAForActionPrediction = get_model(cfg).to("cuda:0").eval()
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
processor = get_processor(cfg)
vla.predict_action = vla.mul_predict_action
image = Image.open("example.png").convert("RGB")
prompt = "In: What action should the robot take to pick the cup?\nOut:"
inputs = processor(text=prompt,images=image).to("cuda:0", dtype=torch.bfloat16)

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