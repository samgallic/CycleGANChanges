DATASET="datasets/just_noise"
DATASET_CLEAN="datasets/black"
NAME="black_debugged_no_disc"

python train.py --dataroot $DATASET --dataroot_clean $DATASET_CLEAN --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 0,1,2,3 --no_flip --emd \
--lambda_noise 1000 --batch_size 16 \
--use_wandb --wandb_project_name cdf --no_img_wandb --n_epochs 200 --n_epochs_decay 200 \
--no_disc 
# --lambda_A 0 --lambda_B 0 --lambda_identity 0

# python test.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
# --no_flip --model cycle_gan --num_test 200 --gpu_ids 0,1
