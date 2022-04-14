CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 main_dino.py \
--optimizer sgd \
--lr 0.03 \
--batch_size_per_gpu 32 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--data_path /home/lijl/Datasets/Imagenet100/train \
--output_dir /home/lijl/Documents/dino/model_saving/surr