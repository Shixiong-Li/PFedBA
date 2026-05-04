# EarlyShield
## Title:
[PAKDD 2026] EarlyShield: Early-Stage Screening for Robust Personalized Federated Learning

## Abstract:
Backdoor attacks pose a serious threat to federated learning (FL). The challenge becomes even more pronounced in personalized FL (PFL), where model updates naturally exhibit high diversity across clients. Existing defenses such as clustering-based detection fail under PFL because benign updates appear highly heterogeneous. What's worse, PFedBA, a recent backdoor on PFL, shows that it can easily bypass most defenses.
To address these limitations, we propose EarlyShield, an effective and data-free defense tailored for both FL and PFL. Our intuition is that even under PFL, benign clients exhibit multi-view consistency, while malicious updates tend to deviate in similarity structure and low-dimensional representations. EarlyShield leverages this idea while focusing on enforcing stringent early screening: (i) client screening based on similarity and principal component analysis (PCA), and (ii) similarity-driven decay to further suppress suspicious updates before aggregation.
Extensive experiments on various datasets across independent and identically distributed (IID) and non-IID settings show that EarlyShield reduces attack success rates with minimal accuracy drop, consistently outperforming existing defenses. We open source the code as well.

## Full paper link:
Will released if published

## Installation
Install Pytorch
## Running
### Generate dataset
Run PFedBA/data/FashionMnist/generate_niid_20users.py
### Run Experiments
- run experiments for the Fashion-MNIST dataset:
  
python main.py --dataset FashionMnist \
               --model dnn \
               --learning_rate 0.1 \
               --numusers 10 \
               --local_epochs 20 \
               --num_global_iters 200 \
               --algorithm pFedMe \
               --per_epoch 1 \
               --poisoning_per_batch 16 \
               --attack_method attackall \
               --attack_start 30 \
               --defense earlyshield \
               --descrip avg_attackall
## Citation:
If you find the code useful in your research, please consider citing our paper:

```
 @inproceedings{li2025earlyshield,
   title={EarlyShield: Early-Stage Screening for Robust Personalized Federated Learning},
   author={Li, Shixiong and Lyu, Xingyu and Wang, Ning and Li, Tao and Chen, Danjue and Hu, Yidan and Chen, Yimin},
   booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
   year={2026},
   organization={Springer}
}
```
        
Note: Our implementation uses parts of some public codes:

[1] PFedBA https://github.com/xtLyu/PFedBA
