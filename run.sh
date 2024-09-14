DATASET="datasets/normal2noisy_forest"
NAME="forest_subtraction_16"

python train.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 2,3 --batch_size 16 --use_wandb --no_flip --emd --wandb_project_name Normal2Noisy \
