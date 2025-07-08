import time
import os
import torch
from PIL import Image
from vla import load_vla

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = load_vla(
    'CogACT/CogACT-Base',
    load_for_training=False,
    action_model_type='DiT-B',
    future_action_window_size=15,
)
model.to('cuda:0').eval()

image = Image.open("example.png").convert("RGB")
prompt = "What action should the robot take to pick the cup?"

for _ in range(5):
    actions, _ = model.predict_action(
        image,
        prompt,
        unnorm_key='fractal20220817_data',
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=10
    )

# torch.cuda.synchronize()
# torch.cuda.reset_peak_memory_stats()
# start = time.time()
# for _ in range(100):
#     actions, _ = model.predict_action(
#         image,
#         prompt,
#         unnorm_key='fractal20220817_data',
#         cfg_scale=1.5,
#         use_ddim=True,
#         num_ddim_steps=10
#     )
# torch.cuda.synchronize()
# end = time.time()

# total = end - start
# avg = total / 100
# peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

# print(f"total time for 100 runs: {total:.4f}s")
# print(f"average time per run: {avg:.4f}s")
# print(f"peak VRAM usage: {peak_vram:.2f} GB")
print(len(actions))
print(actions.shape)
