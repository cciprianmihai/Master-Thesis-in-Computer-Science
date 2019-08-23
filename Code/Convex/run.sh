# run example for resnet
CUDA_VISIBLE_DEVICES=6 python train.py --epochs 30 --batch_size 2048 --learning_rate 0.001 --momentum 0.9 --optimizer lbfgs --save_model
