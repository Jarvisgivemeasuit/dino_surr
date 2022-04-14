# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 eval_linear.py \
CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
--arch resnet50 \
--num_labels 100 \
--batch_size_per_gpu 256 \
--data_path /mnt/share/et21-lijl/Cls_Dataset \
--output_dir /home/et21-lijl/Documents/dino/model_saving/classifier \
--pretrained_weights /home/et21-lijl/Documents/dino/model_saving/cifar10_3208/checkpoint0399.pth