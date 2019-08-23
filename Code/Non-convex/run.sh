# run example for resnet
CUDA_VISIBLE_DEVICES=7 python train.py --epochs 1 --batch_size 2048 --learning_rate 0.1 --momentum 0.9 --net vgg11 --optimizer sgd --num_classes 10 --cifar_type cifar10 --save_model
