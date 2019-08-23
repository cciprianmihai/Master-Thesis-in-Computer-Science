# Import modules
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import time
import shutil
import sys
import multiprocessing
import numpy as np
from torch.autograd import Variable

# Import optimizers
from optimizers import *

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128), only used for train')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='CT', help='sgd, sgd_ls, lbfgs, lbfgs_ls')
parser.add_argument('--line_search', default=None, type=str, metavar='CT', help='backtracking, goldstein, weak_wolfe')
parser.add_argument('--max_samples_per_gpu', default=512, type=int, help='max number of images per GPU')
parser.add_argument('--a_1', default=0.0, type=float, metavar='W', help='alpha_1 (default: 0.0)')
parser.add_argument('--a_2', default=1.0, type=float, metavar='W', help='alpha_2 (default: 1.0)')
parser.add_argument('--tol_grad', default=1e-5, type=float, metavar='W', help='tol_grad (default: 1e-5)')
parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_model', dest='save_model', action='store_true', help='save model')
parser.add_argument('--progress_batches', default=5, type=int, metavar='N', help='number of batches for measure of progress')
parser.add_argument('--store_progress', dest='store_progress', action='store_true', help='store progress of model')

# Define variable for arguments
global args
args = parser.parse_args()

# Redirect output to file
sys.stdout = open('convex_batch'+str(args.batch_size)+'_optimizer'+args.optimizer.upper()+'_epochs'+str(args.epochs)+'_'+str(args.line_search)+'.log', 'w')

print('=> Start building model')
print('=> Python:', sys.version)
print('=> Pytorch:', torch.__version__)
print('=> Numpy:', np.__version__)
print('=> CPUs:', multiprocessing.cpu_count())
print('=> GPUs:', torch.cuda.device_count())
print('=> CUDA:', torch.version.cuda)
print('=> CUDNN:', torch.backends.cudnn.version())
print('=> Arguments:', args)
print('=> Dataset: mnist')
print('=> Model: linear regression')
print('=> Optimizer:', args.optimizer)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train dataset
train_dataset = dsets.MNIST(root ='../../Datasets/mnist/',
                            train = True,
                            transform = transforms.ToTensor(),
                            download = True)
# Test dataset
test_dataset = dsets.MNIST(root ='../../Datasets/mnist/',
                           train = False,
                           transform = transforms.ToTensor())

# Train dataset loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = args.batch_size,
                                           shuffle = True)
# Test dataset loader
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = args.batch_size,
                                          shuffle = False)

# Hyper Parameters
input_size = 784
num_classes = 10

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes).to(device)

# Loss and Optimizer
# Softmax is internally computed
# Set parameters to be updated
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

# Print optimizer parameters
print("Optimizer parameters: ", optimizer.defaults)

# Method for retrieving learning rate
def get_lr(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr

# Train the model
total_step = len(train_loader)

# *** Progress measure ***
# Initial progress step
progress_step = 0
# Sum norm
sum_norm = 0
# ************************

for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

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
        elif args.optimizer == 'sgd_ls' or args.optimizer == 'lbfgs_ls':
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
        # Check if store_progress is True and optimizer is SGD
        if(args.store_progress and (args.optimizer == "sgd_ls" or args.optimizer == "lbfgs_ls")):
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

    # Test the model
    model.eval()
    with torch.no_grad():
        # Calculate accuracy on test set
        correct = 0
        total = 0
        # Iterate through test data set
        for images_test, labels_test in test_loader:
            # Load images to a Torch Variable
            images_test = Variable(images_test.view(-1, 28 * 28))

            images_test = images_test.to(device)
            labels_test = labels_test.to(device)

            # Forward pass only to get output
            outputs = model(images_test)
            # Get prediction
            _, predicted = torch.max(outputs.data, 1)
            # Total number of labels
            total += labels_test.size(0)
            # Total correct predictions
            correct += (predicted == labels_test).sum().item()
        # Compute accuracy
        test_accuracy = 100 * correct / total

    print('Test accuracy: {}_'.format(test_accuracy))

# Save the model checkpoint
if args.save_model:
    torch.save(model.state_dict(), 'convex_batch'+str(args.batch_size)+'_optimizer'+args.optimizer.upper()+'_epochs'+str(args.epochs)+'_'+str(args.line_search)+'.pth')
