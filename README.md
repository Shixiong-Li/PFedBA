# PFedBA

## Installation
Install Pytorch
## Generate dataset
Run PFedBA/data/FashionMnist/generate_niid_20users.py
## Run Experiments
- run experiments for the Fashion-MNIST dataset:
  
python main.py --dataset FashionMnist \
               --model dnn \
               --learning_rate 0.1 \
               --numusers 10 \
               --local_epochs 20 \
               --num_global_iters 150 \
               --algorithm pFedMe \
               --per_epoch 1 \
               --poisoning_per_batch 16 \
               --attack_method attackall \
               --attack_start 30 \
               --defense earlyshield \
               --descrip avg_attackall

