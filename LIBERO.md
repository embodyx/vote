
# Libero Evaluation

## Libero Environment
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

## Libero Evaluation

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
