DATASET="datasets/gamma2rayleigh"
NAME="basic_no_disc"

python train.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 0,1,2,3 --batch_size 16 --emd --no_flip --use_wandb --wandb_project_name cdf \
--no_img_wandb --lambda_A 0 --lambda_B 0 --lambda_identity 0 --no_disc
