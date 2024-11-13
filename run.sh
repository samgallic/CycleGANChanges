DATASET="datasets/gamma2rayleigh"
NAME="test"

python train.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 0,1 --no_flip \
--lambda_noise 1000 --noise_loss_type conditional \
--netD n_layers --n_layers_D 1 --batch_size 16 \
--netG noise 
# --use_wandb --wandb_project_name cdf --no_img_wandb \

# python test.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
# --no_flip --model cycle_gan --num_test 200 --noise_loss_type conditional --gpu_ids 0,1
# --netG noise --netD n_layers --n_layers_D 1
