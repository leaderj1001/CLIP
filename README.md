# CLIP (Contrastive Language–Image Pre-training)

## Experiments (Evaluation)
  | Model | Dataset | Acc (%) |
  |:-:|:-:|:-:|
  | ViT-B/32 (Paper) | CIFAR100 | 65.1 |
  | ViT-B/32 (Our) | CIFAR100 | 61.71 |
  | ViT-B/32 (Paper | CIFAR10 | 91.3 |
  | ViT-B/32 (Our) | CIFAR10 | 88.8 |
  
## Overview
<img width="1333" alt="model" src="https://user-images.githubusercontent.com/22078438/104386323-45c66b00-5578-11eb-9261-0c2c067cabc4.png">


## Training
 - Work In Process

## Usage
 - Evaluation
 ```
 python evaluation.py --dataset CIFAR100 --cuda True
 ```
  - args
    - dataset (str): CIFAR10, CIFAR100 (default: CIFAR100)
    - num_workers (int): default: 0
    - batch_size (int): default: 128
    - cuda (bool): False
  - Training
    - Prepare Data
      - Visual Genome Dataset [link](http://visualgenome.org)
      - Download (images, region descriptions)
    - training
    ```
    python main.py --base_dir ./ --cuda True
    ```
     

## Reference
 - [paper link](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf)
 - Author: Alec Radford, Jong Wook Kim, Chris Hallacy, Girish Sastry, Amanda Askell, Pamela Mishkin, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Jack Clark, Gretchen Krueger, Ilya Sutskever
 - OpenAI
