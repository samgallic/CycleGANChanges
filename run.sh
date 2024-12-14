DATASET="datasets/just_noise"
DATASET_CLEAN="datasets/black"
NAME="histogram_layer_just_noise"

# python train.py --dataroot $DATASET --dataroot_clean $DATASET_CLEAN --name $NAME --input_nc 1 --output_nc 1 \
# --gpu_ids 0,1,2,3 --no_flip --emd --netG hist \
# --lambda_noise 1000 --batch_size 16 \
# --use_wandb --wandb_project_name Histogram-Layer --no_img_wandb \

python train.py --dataroot $DATASET --dataroot_clean $DATASET_CLEAN --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 4,5,6,7 --no_flip --emd --netG hist \
--lambda_noise 1000 --batch_size 16 \
--use_wandb --wandb_project_name Histogram-Layer --no_img_wandb \
--no_disc --lambda_A 0 --lambda_B 0 --lambda_identity 0

# python test.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
# --no_flip --model cycle_gan --num_test 200 --gpu_ids 0,1
