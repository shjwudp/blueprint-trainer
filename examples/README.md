# training gpt2 from sratch

```bash
python train_gpt2.py
```

# data parallel example

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
        data_parallel_training.py \
        --blueprint_filepath ./blueprints/gpt2_blueprint.yaml
```
