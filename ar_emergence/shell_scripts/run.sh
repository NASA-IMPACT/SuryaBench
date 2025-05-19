export CUDA_VISIBLE_DEVICES=0

# python train_baselines.py --config_path ./ds_configs/config_spectformer_ar_sta.yaml --gpu 
python train_baselines.py --config_path ./ds_configs/config_ar_stresnet.yaml --gpu 
