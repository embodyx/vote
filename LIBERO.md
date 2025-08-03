# Libero Evaluation


## Relevant Files

Evaluation
* `experiments/robot/libero/`: LIBERO eval files
    * `run_libero_eval.py`: LIBERO eval script
    * `libero_utils.py`: LIBERO eval utils
    * `batch_eval.py`: Multiple-GPU parallel evaluation script
    * `batch_plot.ipynb`: Plotting script for batch evaluation results
* `experiments/robot/`: General eval utils files
    * `openvla_utils.py`: OpenVLA-specific eval utils
    * `robot_utils.py`: Other eval utils

Training
* `vla-scripts/finetune.py`: VLA fine-tuning script


## Environment
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

## Evaluation


### Multiple GPU evaluation:

```bash
python batch_eval.py  --dir /home/user1/workspace/fle4chunk16token2spatial --devices 0 1  --task_suite libero_spatial
```

The results could be plotted with `batch_plot.ipynb`.

### Single GPU evaluation:
```bash
cd ~/vote/
export PYTHONPATH="$HOME/LIBERO:$PYTHONPATH"
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "juyil/libero_object-b8-3rd_person_img-8act-mul" \
    --task_suite_name "libero_object" \
    --use_proprio False \
    --num_images_in_input 1 \
    --use_l1_regression True \
    --num_actions_chunk 8 \
    --num_actions_per_token 8
```

Chunk16 with two tokens, use:
```bash
    --num_actions_chunk 16 \
    --num_actions_per_token 8
```


## Training

Dataset

```
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```