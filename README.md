# Mamba<sup>&reg;</sup>: Vision Mamba ALSO Needs Registers

#### Official PyTorch implementation of Mamba<sup>&reg;</sup>, a strong vision backbone with pure Mamba blocks

## Release
- [Feb.26.2025] 📢 The paper is accepted by CVPR2025!.

- #### Arxiv: https://arxiv.org/pdf/2405.14858

<img src="Assets\teaser.png" width="80%" />

## Models

|            Model            | IN-1k Accuracy |                          Checkpoint                          |
|:------------------------------------------------------------------:|:-------------:|:----------:|
| Mamba<sup>&reg;</sup>-Small |      81.4      | [mambar_small_patch16_224](https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_small_patch16_224.pth) |
| Mamba<sup>&reg;</sup>-Base  |      83.0      | [mambar_base_patch16_224](https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_base_patch16_224.pth) |
| Mamba<sup>&reg;</sup>-Large |      83.6      | [mambar_large_patch16_224](https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_large_patch16_224.pth) |

* The accuracies are slightly higher than the numbers reported in our manuscript as we improved the training recipe.

## Evaluation

```bash
python -m torch.distributed.launch --nproc_per_node=1  --use_env main.py --model mambar_base_patch16_224 --eval --eval-crop-ratio 1.0 --batch 128 --data-path /PATH/TO/IMAGENET --resume /PATH/TO/CHECKPOINT
```
## Environments

```
# torch>=2.0, cuda>=11.8
pip install timm==0.4.12 mlflow==2.9.1
pip install causal-conv1d==1.1.0
pip install mamba-ssm==1.1.1
```

## Training

Pretrain in 128*128  resolution (single 8-GPU node)

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --model mambar_base_patch16_224 \
  --batch 128 --lr 5e-4 --weight-decay 0.05 \
  --data-path /PATH/TO/IMAGENET \
  --output_dir ./output/mambar_base_patch16_224/pt \
  --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
  --epochs 300 --input-size 128 --drop-path 0.1 --dist-eval
```

Intermediate training in 224*224 resolution

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --model mambar_base_patch16_224 \
  --batch 128 --lr 2e-4 --weight-decay 0.05 \
  --data-path /PATH/TO/IMAGENET \
  --finetune ./output/mambar_base_patch16_224/pt/checkpoint.pth \
  --output_dir ./output/mambar_base_patch16_224/mid \
  --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
  --epochs 100 --input-size 224 --drop-path 0.4 --dist-eval
```

Finetune in 224*224

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --model mambar_base_patch16_224 \
  --batch 64 --lr 1e-5 --weight-decay 0.1 --unscale-lr \
  --data-path /PATH/TO/IMAGENET \
  --finetune ./output/mambar_base_patch16_224/mid/checkpoint.pth \
  --output_dir ./output/mambar_base_patch16_224/ft \
  --reprob 0.0 --smoothing 0.1 --no-repeated-aug \
  --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 \
  --epochs 20 --input-size 224 --drop-path 0.4
```

## Citation
```bibtex
@article{mamba-r,
  title={Mamba-r: Vision mamba also needs registers},
  author={Wang, Feng and Wang, Jiahao and Ren, Sucheng and Wei, Guoyizhe and Mei, Jieru and Shao, Wei and Zhou, Yuyin and Yuille, Alan and Xie, Cihang},
  journal={arXiv preprint arXiv:2405.14858},
  year={2024}
}
```
