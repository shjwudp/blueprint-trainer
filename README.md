# blueprint-trainer

This project provides a scaffolding for sequence model training research.

## About Blueprint

Splitting training design and execution is the core idea of the blueprint-trainer. After decoupling design and execution, you can introduce training testing into the program and ensure your training plan is perfect.

Before the training starts, write your training blueprint in the file. The blueprint looks like this:

```yaml
What It Is:
  GPT2 Training Blueprint

blueprint:
  model: "gpt2"
  optimizer: "AdamW"
  learning_rate:
    base: 1e-4
    scheduler: "inverse_sqrt"
    num_warmup_steps: 1000

  dataset:
  - path: "wikitext"
    name: "wikitext-103-v1"

  batch_size_plan:
  - batch_size: 32
    training_nsteps: 100
  - batch_size: 64
    training_nsteps: 100
  - batch_size: 128
    training_nsteps: 200
  - batch_size: 256
    training_nsteps: 200
  - batch_size: 512
    training_nsteps: -1

  logging:
    path: "./gpt2_training_log"
    interval_by_step: 1
    interval_by_time: "1h"

  checkpoint:
    path: "./gpt2_checkpoints"
    interval_by_step: 1000
    interval_by_time: "1h"

  evaluation:
    interval_by_step: 1000
    interval_by_time: "1h"
```

This blueprint contains all the information required for training but does not include functional behaviors such as model initialization and forward calculation. Users can inject these functions into the trainer through the `trainer.prepare` interface, which is designed to be compatible with various training implementations. For example, you can choose PyTorch Distributed Data Parallel training or a simple single-process training implementation.

## Get Started

The best practice is to try the use cases in the examples directory, and you will find out whether the software is helpful for you.

![3c20cabe940ccde593daf5142a73fcf](https://github.com/shjwudp/blueprint-trainer/assets/11439912/6000fb0c-caad-4f74-a9cc-ead0e6690ab6)
