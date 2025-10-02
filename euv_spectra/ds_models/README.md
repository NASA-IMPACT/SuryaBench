### Example Usage

For training run the below code

1. **Resnet Models**
Multiple resnet models as described below have been used to train the baseline. The only change needed to run the baselines in the given command is to change the model name.
- Resnet18
- Resnet34
- Resnet50
- Resnet101
- Resnet152
```
python train_baseline.py --config_path ./ds_configs/config_resnet_18.yaml --gpu 
```

```
torchrun --nnodes=1 --nproc_per_node=1 train_baseline.py --config_path ./ds_configs/config_resnet_18.yaml --gpu
```

2. **AttentionUNet**
```
python train_baseline.py --config_path ./ds_configs/config_attention_unet.yaml --gpu 
```

```
torchrun --nnodes=1 --nproc_per_node=1 train_baseline.py --config_path ./ds_configs/config_attention_unet.yaml --gpu
```

3. **UNet**
```
python train_baseline.py --config_path ./ds_configs/config_unet.yaml --gpu 
```

```
torchrun --nnodes=1 --nproc_per_node=1 train_baseline.py --config_path ./ds_configs/config_unet.yaml --gpu