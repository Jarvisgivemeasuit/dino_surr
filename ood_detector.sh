CUDA_VISIBLE_DEVICES=1 python ood_detector.py \
--arch resnet50 \
--num_labels 100 \
--batch_size_per_gpu 32 \
--data_path /home/et21-lijl/Datasets/Imagenet100 \
--output_dir /home/et21-lijl/Documents/dino/model_saving/cifar10_classifier \
--pretrained_weights /home/et21-lijl/Documents/dino/model_saving/imagenet100/checkpoint0399.pth