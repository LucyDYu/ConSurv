# Run experiments simultaneously with 2 GPUs
# If you only have one GPU available, you may delete the "&" sign at the end of each line

# Joint Training
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --n_epochs=20 --model_config=best --model=sgd --dataset_config=joint;

# SGD
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --n_epochs=20 --model_config=best --model=sgd &

# EWC
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --n_epochs=20 --model_config=best --model=ewc_on &

# LwF
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=lwf --model_config=best --n_epochs=20 &

# DER
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=der --model_config=default --buffer_size=32 --n_epochs=20 &

# DERPP
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=derpp --model_config=default --buffer_size=32 --n_epochs=20 &

# ER
CUDA_VISIBLE_DEVICES=1 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=er --model_config=default --buffer_size=32 --n_epochs=20 &

# IMEX-reg
CUDA_VISIBLE_DEVICES=1 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=imex_reg --model_config=default --buffer_size=32 --n_epochs=20 &

# MOSE
CUDA_VISIBLE_DEVICES=1 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=mose_wsi --model_config=default --buffer_size=32 --n_epochs=20 &

---
# ConSurv (ours), MS MOE + FCR
CUDA_VISIBLE_DEVICES=0 python3 main_wsi.py --wandb_entity=<your-wandb-entity> --wandb_project=<your-wandb-project> --fold=0 --model=consurv --model_config=default --buffer_size=32 --n_epochs=20 &
