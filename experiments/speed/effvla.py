import os
import time
import torch

from experiments.robot.openvla_utils import get_action_head, get_processor,GenerateConfig
from experiments.robot.robot_utils import get_model
from PIL import Image

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# one token
# cfg = GenerateConfig(
#     pretrained_checkpoint="juyil/libero_object-b8-3rd_person_img-8act-mul",
#     use_l1_regression=True,
#     use_diffusion=False,
#     use_film=False,
#     num_images_in_input=1,
#     use_proprio=False,
#     load_in_8bit=False,
#     load_in_4bit=False,
#     center_crop=True,
#     num_open_loop_steps=8,
#     unnorm_key="libero_object_no_noops",
#     mode="mul",
#     num_actions_chunk=8,
#     num_actions_per_token=8,
#     action_head_name="mlp",
#     num_blocks=2,
# )


# two tokens
cfg = GenerateConfig(
    pretrained_checkpoint="juyil/spatial-b8-16act-2token-60ksteps",
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
    num_actions_chunk=16,
    num_actions_per_token=8,
    action_head_name="mlp",
    num_blocks=2,
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
