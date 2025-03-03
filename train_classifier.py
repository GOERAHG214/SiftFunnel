from __future__ import print_function
import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
import sys
import time
import torch.nn.functional as F
from target_model_packages import *
from tqdm import tqdm
from utilis import *
from torchsummary import summary
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_time = datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='')
parser.add_argument('--epochs', type=int, default=200, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--allocation', type=float, default=0.8, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nz', type=int, default=2)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=6, metavar='')
parser.add_argument('--dataset', default='chest', metavar='')
parser.add_argument('--target_model', default='cnn', help='cnn | vgg | ir152 | resnet')
parser.add_argument('--optimize', default='adam', help='sgd | adam')
parser.add_argument('--scheduler', default='no', help='yes | no')
parser.add_argument('--user', default='lenovo', help='yes | no')


# Training records
class Logger(object):
    def __init__(self, filename="mnist_classifier.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# Training Classifier
def train(classifier, log_interval, device, data_loader, optimizer, epoch, dataset_size):
    # import setting and classifier
    args = parser.parse_args()
    classifier.train()

    # Training time and accuracy object initialization
    T = 0
    correct = 0

    # Training set iteration
    for batch_idx, (data, target) in enumerate(data_loader):
        T1 = time.process_time()

        # Import training data and labels
        data, target = data.to(device), target.to(device)

        # Clear the gradient, get intermediate features with the prediction
        optimizer.zero_grad()
        feat, output = classifier(data, release=False)

        # Calculation of loss
        loss = F.nll_loss(output, target.long())

        # Get predicted labels through prediction and calculate the exact quantity
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Get the gradient and optimize the classifier
        loss.backward()
        optimizer.step()

        T2 = time.process_time()
        T += T2 - T1

        # A loss will be output when the preset interval is reached.
        # Update tqdm progress bar
        data_loader.set_postfix(loss=loss.item(), accuracy=100. * correct / (dataset_size))
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
        #                                                          len(data_loader.dataset), loss.item()))

    # Output consumed time and training set accuracy after finishing an epoch round
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, Learning Rate: {current_lr}')
    print('Each round takes:{0:.4f}s {1:.4f}min'.format(T, T / 60))
    print('Accuracy: {}/{} ({:.4f}%)'.
          format(correct, dataset_size, 100. * correct / dataset_size))
    return T


def test(classifier, device, data_loader):
    # Import the trained classifier and enable evaluation mode
    args = parser.parse_args()
    classifier.eval()

    # Initialize test loss and accuracy objects
    test_loss = 0
    correct = 0

    # There will be no gradient computation in the test and iterated in the test set
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            _, output = classifier(data, release=False)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset), test_loss


def main():
    # Initialize the log folder and export the records of this training
    args = parser.parse_args()
    os.makedirs(f'log/{args.dataset}', exist_ok=True)
    sys.stdout = Logger(f'./log/{args.dataset}/{args.dataset}_{args.target_model}_{args.lr}_train_log.txt')
    print("Formatted time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("================================")
    print(args)
    print("================================")
    os.makedirs('out', exist_ok=True)

    # Compute using GPU
    total_time = 0
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    # Random seed initialization, temporarily unavailable
    torch.manual_seed(args.seed)
    random.seed(666)

    # Define the preprocessor for input data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
    transform_32 = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    transform_amplify = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                   hue=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])
    transform_chest = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                 hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                          ])

    # Define the preprocessing transformations for the test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # Normalize the image with mean and std deviation
    ])
    test_chest = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,))
        # Normalize the image with mean and std deviation
    ])

    # The definition of target model
    if args.target_model == "cnn":
        classifier = Classifier_4(nc=args.nc, ndf=args.ndf, nz=args.nz, img_size=args.img_size).to(device)
        summary(classifier, (args.nc, args.img_size, args.img_size))
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
    elif args.target_model == "resnet":
        classifier = rn18(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to(device)
        summary(classifier, (args.nc, args.img_size, args.img_size))
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
    elif args.target_model == 'vgg':
        classifier = vgg16_bn(nc=args.nc, nz=args.nz, img_size=args.img_size).to(device)
        summary(classifier, (args.nc, args.img_size, args.img_size))
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.85, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True,
                                   weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                   verbose=True)
        elif args.optimize == "adamw":
            optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True,
                                    weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                                   verbose=True)
    elif args.target_model == "ir152":
        classifier = IR152(args.nz, args.nc, args.img_size).to(device)
        summary(classifier, (args.nc, args.img_size, args.img_size))
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.85, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True,
                                   weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180], gamma=0.5)
        elif args.optimize == "adamw":
            optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, betas=(0.6, 0.999), amsgrad=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                             verbose=True)

    # Get the model's state dictionary
    state_dict = classifier.state_dict()

    # Calculate number of parameters and total size
    param_count = sum(p.numel() for p in state_dict.values())
    param_size_mb = param_count * 4 / (1024 ** 2)  # Convert to MB

    print(f"Total number of parameters: {param_count}")
    print(f"Model size (MB): {param_size_mb:.2f} MB")

    # The definition of data set
    if args.dataset == 'facescrub':
        train_set = FaceScrub(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform_amplify, train=True)
        test_set = FaceScrub(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=test_transform, train=False)

    elif args.dataset == 'chest':
        train_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform_chest, train=True)
        test_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=test_chest, train=False)

    elif args.dataset == 'cifar10':
        train_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', train=True,
                               transform=transform_amplify, )
        test_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', train=False, transform=test_transform)

    # Load the data set into memory
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Object initialization with optimal number of iteration rounds and accuracy
    best_cl_acc = 0
    best_cl_epoch = 0
    cl_acc_lst, loss_lst = [], []
    dataset_size = len(train_loader.dataset)

    # Train classifier
    for epoch in range(1, args.epochs + 1):
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100, leave=True)
        total_time += train(classifier, args.log_interval, device, train_loader, optimizer, epoch, dataset_size)
        cl_acc, loss = test(classifier, device, test_loader)
        if args.scheduler == "yes":
            scheduler.step(loss)
        cl_acc_lst.append(cl_acc)
        loss_lst.append(loss)

        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'best_cl_acc': best_cl_acc,
            }
            os.makedirs(f'./target_model/{args.dataset}', exist_ok=True)
            torch.save(state, f'./target_model/{args.dataset}/{args.dataset}_{args.target_model}_{args.lr}.pth')

    print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))
    print('Take:{0:.4f}s {1:.4f}min'.format(total_time, total_time / 60))

    print("Formatted time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    # Plot a line chart of the number of iterations versus loss and accuracy
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(range(args.epochs), loss_lst, label='Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='red')
    ax2.plot(range(args.epochs), cl_acc_lst, label='Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Training Loss and Accuracy')

    fig.tight_layout()
    plt.savefig(f'./target_model/{args.dataset}_{args.target_model}_{args.lr}.png')
    plt.show()


# Initiation!
if __name__ == '__main__':
    main()
