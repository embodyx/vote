import os
import time
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda:6"
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

image = Image.open("example.png").convert("RGB")
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

for _ in range(5):
    _ = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)


torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

torch.cuda.synchronize()
end = time.time()
total = end - start
avg = total / 100
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"total time for 100 runs: {total:.4f}s")
print(f"average time per run: {avg:.4f}s")
print(f"peak VRAM usage: {peak_vram:.2f} GB")
print(action)
