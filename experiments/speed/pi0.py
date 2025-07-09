
import time
import jax

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
import numpy as np

config = _config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example =  {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "What action should the robot take to pick the cup?",
    }

# Warmup runs
print("Running warmup...")
for _ in range(5):
    result = policy.infer(example)

# Reset timing and run benchmark
print("Running benchmark...")
start = time.time()

for _ in range(100):
    result = policy.infer(example)

jax.block_until_ready(result)
end = time.time()

total = end - start
avg = total / 100

print(f"total time for 100 runs: {total:.4f}s")
print(f"average time per run: {avg:.4f}s")

#patch the following code  in  src/openpi/policies/droid_policy.py to adapt one image 
# images = (base_image, np.zeros_like(base_image))
# image_masks = (np.True_, np.True_)
# names = ("base_0_rgb", "base_1_rgb")
