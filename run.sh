DATASET="datasets/normal2noisy_forest"
NAME="print"

python train.py --dataroot $DATASET --name $NAME --input_nc 1 --output_nc 1 \
--gpu_ids 0,1 --batch_size 16 --use_wandb --no_flip --emd --wandb_project_name Normal2Noisy \
