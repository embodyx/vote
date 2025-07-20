# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting

**Paper: https://arxiv.org/abs/2507.05116**


 [\[🤗 Model Zoo\]](https://huggingface.co/collections/juyil/vote-vision-language-action-model-686f5dac2775080477a86cdf) 

## News 
- `2025/07/10`: We release the [Vote 1.0](https://huggingface.co/collections/juyil/vote-vision-language-action-model-686f5dac2775080477a86cdf).


# Installation


```bash
conda create -n effvla python=3.10 -y
conda activate effvla

cd ~/ 
git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install flash-attn==2.6.1 --no-build-isolation
```


### Quick start

```bash
cd experiments/speed/
python effvla.py
```


### Libero Environment
```bash
cd ~/  
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

then install libero requirement.
```bash
cd ~/vote
pip install -r experiments/robot/libero/libero_requirements.txt 
```

### Speed

We have speed measurement codes under `experiments/speed/`. 

# Training

## Training Setting

- **LIBERO**: NVIDIA RTX A6000 GPUs (48 GB VRAM each), 256 GB system RAM, shuffle buffer of 60K samples. Global batch size 40. Training converges in under 24 hours.
- **Fractal**: NVIDIA H100 NVL GPUs (94 GB VRAM each), 756 GB system RAM, shuffle buffer of 256K samples. Global batch size 144. Training converges in under 24 hours.
- **Bridge**: NVIDIA H100 NVL GPUs (94 GB VRAM each), 756 GB system RAM, shuffle buffer of 256K samples. Global batch size 96. Training converges in under 24 hours.


## Steps

Dataset follows: [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod)

Then run train script:
```
bash train.sh
```


### Q&A

For libero, read `LIBERO.md` carefully.


```
No module named prismatic
No module named experiments
```

It is because you don't install correctly. Check `pip list | grep effvla`. 

If you run into any issues, please open a new GitHub issue.



# SimplerEnv
For SimplerEnv installation:

You may install SimplerEnv before you install effvla.
Because install tensorflow 2.15 will break the cuda env in torch.


```bash
conda create -n simpler_env python=3.10
conda activate simpler_env

git clone  https://github.com/LukeLIN-web/simplerenv.git --recurse-submodules
pip install numpy==1.24.4 # important, numpy >=1.26 has problem in simpler env

cd simplerenv/ManiSkill2_real2sim
pip install -e .

cd simplerenv
pip install -e .

git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip install -e .

sudo apt install ffmpeg

cd simplerenv
pip install tensorflow==2.15.0
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support


# we  need to install torch and torchvision again if it shows libtorch_cuda.so: undefined symbol: ncclCommRegister 
pip install torch==2.3.1 torchvision==0.18.1
pip install mediapy pandas
pip install gymnasium==0.28.1
```

### Results

Evaluation results on the WidowX robot in the SimplerEnv Visual Matching setting. 

| Method                  | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg. | Latency (ms) ↓ | Speed up ↑ |
|-------------------------|-----------|------------|--------------|---------------|------|----------------|-------------|
| RT-1-X                  | 0.0       | 4.2        | 0.0          | 0.0           | 1.1  | --             | --          |
| Octo             | 47.2      | 9.7        | 4.2          | 56.9          | 30.0 | --             | --          |
| OpenVLA                 | 0.0       | 0.0        | 0.0          | 4.1           | 1.0  | 240            | 1.00        |
| RoboVLM    | 29.2      | 25.0       | 12.5         | 58.3          | 31.3 | --             | --          |
| Openpi0                 | 29.1      | 0.0        | 16.6         | 62.5          | 27.1 | 470            | 0.50        |
| SpatialVLA | 16.7      | 25.0       | 29.2         | 100.0         | 42.7 | 400            | 0.60        |
| CogACT                  | 71.7      | 50.8       | 15.0         | 67.5          | 51.3 | 220            | 1.09        |
| __Ours__               | __54.2__  | __25.0__   | __45.8__     | __91.7__      | __54.2__ | __78__     | __3.07__    |


# TODO

- [ ] Upload all LIBERO checkpoints  
- [ ] Upload SimplerEnv evaluation code and checkpoints  
- [ ] Upload training scripts



## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2507.05116):

```bibtex
@misc{lin2025votevisionlanguageactionoptimizationtrajectory,
      title={VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting}, 
      author={Juyi Lin and Amir Taherin and Arash Akbari and Arman Akbari and Lei Lu and Guangyu Chen and Taskin Padir and Xiaomeng Yang and Weiwei Chen and Yiqian Li and Xue Lin and David Kaeli and Pu Zhao and Yanzhi Wang},
      year={2025},
      eprint={2507.05116},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05116}, 
}
```