# Import modules
import argparse
import os
import time
import shutil
import sys
import multiprocessing
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Import optimizers
from optimizers import *

# Import models
from models import *

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128), only used for train')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--cifar_type', default='cifar10', type=str, metavar='CT', help='cifar10, cifar100 (default: cifar10)')
parser.add_argument('--net', default='resnet20', type=str, metavar='CT', help='resnet20')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='CT', help='sgd, sgd_ls, lbfgs, lbfgs_ls, lbfgs_ls_inf')
parser.add_argument('--line_search', default=None, type=str, metavar='CT', help='backtracking, goldstein, weak_wolfe')
parser.add_argument('--num_classes', default='10', type=int, metavar='CT', help='10, 100 (default)')
parser.add_argument('--max_samples_per_gpu', default=512, type=int, help='max number of images per GPU')
parser.add_argument('--a_1', default=0.0, type=float, metavar='W', help='alpha_1 (default: 0.0)')
parser.add_argument('--a_2', default=1.0, type=float, metavar='W', help='alpha_2 (default: 1.0)')
parser.add_argument('--tol_grad', default=1e-5, type=float, metavar='W', help='tol_grad (default: 1e-5)')
parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_model', dest='save_model', action='store_true', help='save model')
parser.add_argument('--progress_batches', default=5, type=int, metavar='N', help='number of batches for measure of progress')
parser.add_argument('--store_progress', dest='store_progress', action='store_true', help='store progress of model')
parser.add_argument('--valid_size', default=0.2, type=float, metavar='W', help='valid_size (default: 0.2)')

# Define variable for arguments
global args
args = parser.parse_args()

# Redirect output to file
sys.stdout = open(args.net+'_'+str(args.cifar_type)+'_batch'+str(args.batch_size)+'_optimizer'+args.optimizer.upper()+'_epochs'+str(args.epochs)+'_'+str(args.line_search)+'.log', 'w')

print('=> Start building model')
print('=> Python:', sys.version)
print('=> Pytorch:', torch.__version__)
print('=> Numpy:', np.__version__)
print('=> CPUs:', multiprocessing.cpu_count())
print('=> GPUs:', torch.cuda.device_count())
print('=> CUDA:', torch.version.cuda)
print('=> CUDNN:', torch.backends.cudnn.version())
print('=> Arguments:', args)
print('=> Dataset: ', args.cifar_type)
print('=> Model:', args.net)
print('=> Optimizer:', args.optimizer)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
if args.cifar_type == 'cifar10':
    train_dataset = torchvision.datasets.CIFAR10(root='../../Datasets/cifar10/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='../../Datasets/cifar10/',
                                                train=False,
                                                transform=transforms.ToTensor())
# CIFAR-100 dataset
elif args.cifar_type == 'cifar100':
    train_dataset = torchvision.datasets.CIFAR100(root='../../Datasets/cifar100/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR100(root='../../Datasets/cifar100/',
                                                train=False,
                                                transform=transforms.ToTensor())

# Prepare indices for train_loader and validation_loader
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(args.valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)

# Model
if args.net == 'resnet20':
    model = resnet20(num_classes = args.num_classes).to(device)
elif args.net == 'vgg11':
    model = vgg11().to(device)
elif args.net == 'vgg11_bn':
    model = vgg11_bn().to(device)
elif args.net == 'vgg13':
    model = vgg13().to(device)
elif args.net == 'vgg13_bn':
    model = vgg13_bn().to(device)
elif args.net == 'vgg16':
    model = vgg16().to(device)
elif args.net == 'vgg16_bn':
    model = vgg16_bn().to(device)
elif args.net == 'vgg19':
    model = vgg19().to(device)
elif args.net == 'vgg19_bn':
    model = vgg19_bn().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
elif args.optimizer == 'sgd_ls':
    optimizer = SGD(model.parameters(),
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    line_search=args.line_search,
                    tolerance_grad=args.tol_grad,
                    a_1=args.a_1,
                    a_2=args.a_2)
elif args.optimizer == 'lbfgs':
    optimizer = torch.optim.LBFGS(model.parameters(),
                    lr=args.learning_rate,
                    line_search_fn=args.line_search,
                    tolerance_grad=args.tol_grad,
                    max_iter=10)
elif args.optimizer == 'lbfgs_ls':
    optimizer = LBFGS(model.parameters(),
                    lr=args.learning_rate,
                    line_search_fn=args.line_search,
                    tolerance_grad=args.tol_grad,
                    max_iter=10,
                    a_1=args.a_1,
                    a_2=args.a_2)
elif args.optimizer == 'lbfgs_ls_inf':
    optimizer = LBFGS2(model.parameters(),
                    lr=args.learning_rate,
                    line_search_fn=args.line_search,
                    tolerance_grad=args.tol_grad,
                    max_iter=10)
# Print optimizer parameters
print("Optimizer parameters: ", optimizer.defaults)

# Method for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Method for retrieving learning rate
def get_lr(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr

# Train the model
total_step = len(train_loader)
learning_rate = args.learning_rate

# *** Progress measure ***
# Initial progress step
progress_step = 0
# Sum norm
sum_norm = 0
# ************************

for epoch in range(args.epochs):
    # Train the model
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Define the closure for optimizer
        def closure():
            # Clear gradients, not be accumulated
            optimizer.zero_grad()
            # Forward pass to get output
            outputs = model(images)
            # Calculate loss: softmax + cross entropy loss
            loss = criterion(outputs, labels)
            # Get gradients
            loss.backward()
            # Return loss
            return loss

        # Get lr and loss
        if args.optimizer == 'sgd' or args.optimizer == 'lbfgs':
            lr = get_lr(optimizer)
            loss = optimizer.step(closure)
        elif args.optimizer == 'sgd_ls' or args.optimizer == 'lbfgs_ls' or args.optimizer == 'lbfgs_ls_inf':
            loss, lr = optimizer.step(closure)

        # Compute accuracy for the model
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total

        args.print_freq = int(total_step / 4)
        if args.print_freq == 0:
            args.print_freq = 1

        if i == 0 or (i + 1) % args.print_freq == 0:
            print ("Epoch: [{}][{}/{}]_, Loss: {:.4f}_ Accuracy: {:.2f}_ Learning rate: {:.5f}"
                   .format(epoch, i, total_step, loss.item(), train_accuracy, lr))

        # *** Progress measure ***
        # Update progress step
        progress_step += 1
        # Check if store_progress is True and optimizer is SGD with LS / LBFGS with LS
        if(args.store_progress and (args.optimizer == "sgd_ls" or args.optimizer == "lbfgs_ls" or args.optimizer == "lbfgs_ls_inf")):
            # Initial step
            if (progress_step == 1):
                # Save initial model
                initial_model = optimizer._gather_flat_data()
            # Save current grad norm
            current_grad_norm = optimizer._grad_norm()
            # Sum of the norms
            sum_norm += current_grad_norm
            # Final step
            if (progress_step % args.progress_batches == 0):
                # Save final model
                final_model = optimizer._gather_flat_data()
                # Subtract from initial model the final model
                diff_model = final_model.sub(initial_model)
                # Compute the norm-2 of difference
                diff_norm_model = diff_model.norm()
                # Print the progress
                print("After", progress_step, "steps: ")
                result = sum_norm / diff_norm_model
                print("Progress:", result.item())
                # Reset norms array
                sum_norm = 0
                # Reset progress step
                progress_step = 0
        # ************************

    # Test the model on validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images_valid, labels_valid in valid_loader:
            images_valid = images_valid.to(device)
            labels_valid = labels_valid.to(device)
            outputs = model(images_valid)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_valid.size(0)
            correct += (predicted == labels_valid).sum().item()
        valid_accuracy = 100 * correct / total

        print('Validation accuracy: {}_'.format(valid_accuracy))

    # Test the model on test set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images_test, labels_test in test_loader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)
            outputs = model(images_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()
        test_accuracy = 100 * correct / total

        print('Test accuracy: {}_'.format(test_accuracy))

    # Decay learning rate
    if (epoch+1) == 80 or (epoch+1) == 120 or  (epoch+1) == 160:
        learning_rate /= 10
        update_lr(optimizer, learning_rate)

# Save the model checkpoint
if args.save_model:
    torch.save(model.state_dict(), args.net+'_'+str(args.cifar_type)+'_batch'+str(args.batch_size)+'_optimizer'+args.optimizer.upper()+'_epochs'+str(args.epochs)+'_'+str(args.line_search)+'.pth')
