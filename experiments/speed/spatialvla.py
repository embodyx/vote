import time
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_name_or_path = "IPEC-COMMUNITY/spatialvla-4b-224-pt"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()

image = Image.open("example.png").convert("RGB")
prompt = "What action should the robot take to pick the cup?"
inputs = processor(images=[image], text=prompt, return_tensors="pt")

for _ in range(5):
    generation_outputs = model.predict_action(inputs)
    actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    generation_outputs = model.predict_action(inputs)
    actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
torch.cuda.synchronize()
end = time.time()

peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
total = end - start
print(f"total time for 100 runs: {total:.4f}s")
print(f"average time per run: {total/100:.4f}s")
print(f"peak VRAM usage: {peak_vram:.2f} GB")
print(actions)
