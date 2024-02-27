# LoraCaptioner

## Description
A simple image captioning model fine-tuned using LoRA with Conceptual Captions dataset.

- Image encoder: ViT-B/16 (using slightly modified version of openai/clip)
- Text decoder: Flan-T5-base (using 🤗 transformers)
- LoRA: Attached to text decoder (using 🤗 peft)
- Data: Used first 300K (unshuffled) rows from Conceptual Captions dataset

[Wandb workspace](https://wandb.ai/wittgensteinian/tl_summer2023?workspace=user-wittgensteinian)


## Results (not cherry-picked)

| ![Image](results/100.jpg) |
|:-------------------------:|
|     <b> Caption: </b>     |
|   logo of the ad agency   |
|       <b> Id: </b>        |
|            100            |

|  ![Image](results/102.jpg)  |
|:---------------------------:|
|      <b> Caption: </b>      |
| a sand castle in the desert |
|        <b> Id: </b>         |
|             102             |

|                                                                         ![Image](results/103.jpg)                                                                          |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                             <b> Caption: </b>                                                                              |
| tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical tropical |
|                                                                                <b> Id: </b>                                                                                |
|                                                                                    103                                                                                     |

|       ![Image](results/101.jpg)        |
|:--------------------------------------:|
|           <b> Caption: </b>            |
| sand dunes and sand dunes on the beach |
|              <b> Id: </b>              |
|                  101                   |

### Setup
- Used checkpoint `abkx98e7`, trained on 300K images for 8 epochs (lr=1e-4).
- Used greedy decoding

### Results
- Caption generally captures the overall semantics of image.
- The model often fails to generate fluent sentences, a classic problem of text degeneration.
