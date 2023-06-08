# Technical Interview Task 2

## Output 1: loss curve
[Wandb workspace](https://wandb.ai/wittgensteinian/tl_summer2023?workspace=user-wittgensteinian)

### Best model
- Training loss falls; so does validation loss.
- Training loss seems to converge (does not fall anymore) even before reaching a single epoch.

### Ablation study on modules to which LoRA is applied
- Removing LoRA from FFN layers (so LoRA only applied on attention layers) is bad. This is interesting as original LoRA paper only applied LoRA to attention layers.

## Output 2: captioning results
Still working on it.  
I have the trained model, but don't have the method to generate text.

## Output 3: codebase
This repo itself.


## Appendix

### Dashboard

### Cross-attention between image and text
