import argparse

from collections import OrderedDict
from models import *
import torch
import torch.nn as nn
import time
from torchvision import datasets, models
from torchvision import transforms

# Import models
from models.resnet import *


def _bn_train_mode(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def hess_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    if cuda:
        vector = vector.to("cuda")
    param_list = list(model.parameters())
    vector_list = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input, target = input.to("cuda"), target.to("cuda")
        output = model(input)
        loss = criterion(output, target)
        loss *= input.size()[0] / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)

        dL_dvec = torch.zeros(1)
        if cuda:
            dL_dvec = dL_dvec.cuda()
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)

        dL_dvec.backward()
    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)

criterion = nn.CrossEntropyLoss()

class HessVecProduct(object):
    def __init__(self, loader, model, criterion, device):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.device = device
        self.iters = 0
        self.timestamp = time.time()

    def __call__(self, vector):
        start_time = time.time()
        output = hess_vec(
            vector,
            self.loader,
            self.model,
            self.criterion,
            cuda=(self.device == "cuda"),
            bn_train_mode=True,
        )
        time_diff = time.time() - start_time

        self.iters += 1
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        return output.to("cpu").unsqueeze(1)


def prepareLoader(mode, dataset):
    train = mode == "train"
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root="../../Datasets/cifar10", train=train,
            download=True, transform=transform,
        )
    elif dataset == "cifar100":
        dataset = datasets.CIFAR100(
            root="../../Datasets/cifar100", train=train,
            download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True
    )
    return data_loader


def prepareModel(model_name, model_file, dataset):
    num_classes = 10 if dataset == "cifar10" else 1000
    model = None
    if model_name == "resnet20":
        model = resnet20(num_classes=num_classes)

    model.load_state_dict(torch.load(model_file))

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device):
    correct = 0
    loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss += criterion(output, target).item() * input.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Results on", args.mode, "mode")
    print("Accuracy %0.2f" % (100 * correct / len(loader.dataset)))
    print("Loss %0.5f" % (loss / len(loader.dataset)))


parser = argparse.ArgumentParser(description="Eigenvalues finder")

parser.add_argument("--model_name", default="resnet20", type=str, help="Model (default: resnet20)")
parser.add_argument("--model_file", default="checkpoints.pth", type=str, help="Model file (default: checkpoint.pth)")
parser.add_argument("--device", default="cuda", type=str, help="Device (default: cuda)")
parser.add_argument("--iterations", default=1000, type=int, help="Iterations (default: 1000)")
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset (default: cifar10)")
parser.add_argument("--mode", default="train", type=str, help="Mode to run on (default: train")
parser.add_argument("--file", default="output.txt", type=str, help="File to save the eigenvalues (default: output.txt)")
parser.add_argument("--evaluate", action='store_true', help="Evaluate the model (default: False)")
args = parser.parse_args()

loader = prepareLoader(args.mode, args.dataset)
model = prepareModel(args.model_name, args.model_file, args.dataset).to(args.device)

if args.evaluate:
    evaluate(model, loader, args.device)
else:
    print(model)
    num_parameters = count_parameters(model)
    print("Number of parameters", num_parameters)

    productor = HessVecProduct(loader, model, criterion, args.device)

    print(args.iterations)
    print((num_parameters, num_parameters)[-1])
    Q, T = lanczos_tridiag(productor, args.iterations, dtype=torch.float32, device="cpu",
                           matrix_shape=(num_parameters, num_parameters))

    eigvals, eigvects = T.eig(eigenvectors=True)

    eigvals = eigvals.to("cpu")

    output = open(args.file, "w")
    for val in eigvals:
        output.write("{}\n".format(val[0].item()))
