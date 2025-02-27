# Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement

### Requirements

The following environment and dependencies are required:

- **GPU**: RTX 4060 with 8GB VRAM is sufficient
- **Libraries**:
  - `timm` version 0.9.12
  - `torch` version 1.12.1+cu116
  - `torchvision` version 0.13.1+cu116
  - `numpy` version 1.24.4

### Running the Attack and Evaluation

For example, to run the **FPR+GRA (GRA with 5 samples per iteration)** attack, execute the following command:

```
CUDA_VISIBLE_DEVICES=0 python main.py --attack vitb_gra
```

For **evaluation**, you can run:

```
CUDA_VISIBLE_DEVICES=0 python main.py --eval
```

### Hyperparameter Tuning

When working with different datasets, you can achieve better results by fine-tuning the current hyperparameters. Experimenting with various hyperparameter settings based on the specific characteristics of the dataset may help improve the performance of the attack.

### Code References

We would like to express our gratitude to the previous researchers for their selfless contributions. Our code heavily benefits from [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack).
