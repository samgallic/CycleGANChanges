DATASET="datasets/gamma2rayleigh"
NAME="hist_lambda_ten"

python train.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 0,1,2,3 --batch_size 16 --no_flip --emd --use_wandb --wandb_project_name Gamma2Rayleigh
