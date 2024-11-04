DATASET="datasets/less_noise"
NAME="less_noise_32"

python train.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 0,1,2,3 --batch_size 32 --emd --no_flip --use_wandb --wandb_project_name cdf \
--no_img_wandb --lambda_noise 1000 \
# --netG noise --netD n_layers --n_layers_D 1