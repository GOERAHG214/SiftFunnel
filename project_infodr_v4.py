from __future__ import print_function
import argparse
import random
import re
import matplotlib.pyplot as plt
import os

from torch.nn.functional import l1_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
import sys
from hsic import *
from ptflops import get_model_complexity_info as summary_info
import time
from T_model_packages import *
import torch.nn.functional as F

from InfoDR_package import *
from target_model_packages import *
from tqdm import tqdm
from utilis import *
from torchsummary import summary
from datetime import datetime


current_time = datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='')
parser.add_argument('--epochs', type=int, default=300, metavar='')
parser.add_argument('--epochs_vae', type=int, default=300, metavar='')
parser.add_argument('--epochs_cloud', type=int, default=300, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--smooth_factor', type=float, default=0.25, metavar='')
parser.add_argument('--vae_lr', type=float, default=0.001, metavar='')
parser.add_argument('--alpha', type=float, default=5, metavar='')
parser.add_argument('--beta', type=float, default=2, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--allocation', type=float, default=0.8, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--block_idx', type=int, default=1)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--gamma', type=float, default=2, metavar='')
parser.add_argument('--nz', type=int, default=2)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=6, metavar='')
parser.add_argument('--dataset', default='chest', metavar='')
parser.add_argument('--target_model', default='cnn', help='cnn | vgg | ir152 | resnet')
parser.add_argument('--optimize', default='adam', help='sgd | adam')
parser.add_argument('--user', default='lenovo', help='sgd | adam')
parser.add_argument('--scheduler', default='no', help='yes | no')


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


def get_pth_paths(base_dir, dataset, target_model):
    full_path = None
    pattern = re.compile(rf'{re.escape(dataset)}_(?P<target_model>.+?)\.pth')

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pth"):
                match = pattern.match(file)
                if match:
                    file_target_model = match.group('target_model')
                    if file_target_model.startswith(target_model):
                        full_path = os.path.join(root, file)
                        return full_path


def get_estimator_paths(base_dir, dataset, target_model, block_idx):
    full_path = None
    target_file = f"inversion_{block_idx}.pth"

    dataset_dir = os.path.join(base_dir, dataset)

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        return None

    for model_dir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model_dir)

        if os.path.isdir(model_path) and model_dir.startswith(target_model):
            full_path = os.path.join(model_path, target_file)

            if os.path.exists(full_path):
                return full_path

    print(f"No matching .pth file found for dataset '{dataset}' and target_model '{target_model}' with block_idx {block_idx}.")
    return None


def label_smooth(target, num_classes, smooth_factor):
    with torch.no_grad():
        smoothed_labels = torch.full((target.size(0), num_classes), smooth_factor / (num_classes - 1),
                                     device=target.device)
        smoothed_labels.scatter_(1, target.unsqueeze(1), 1.0 - smooth_factor)  # Translated comment 1 - smooth_factor
    return smoothed_labels


def label_smoothing_cross_entropy_loss(probs, targets, smoothing=0.2):
    args = parser.parse_args()
    num_classes = args.nz

    one_hot_targets = torch.zeros_like(probs).scatter(1, targets.view(-1, 1), 1)

    cross_entropy_loss_one_hot = -torch.sum(one_hot_targets * torch.log(probs + 1e-9), dim=1)

    all_ones_target = torch.ones_like(probs) / num_classes
    cross_entropy_loss_all_ones = -torch.sum(all_ones_target * torch.log(probs + 1e-9), dim=1)

    # 5. 合并两部分损失
    final_loss = (1 - smoothing) * F.cross_entropy(probs, targets) + (smoothing / num_classes) * cross_entropy_loss_all_ones

    # final_loss = torch.mean(-torch.sum(targets * probs, dim=-1))
    return final_loss

def orthogonality_loss(features):
    b, c, _, _ = features.size()
    normalized_features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
    inner_product = torch.bmm(normalized_features.view(b, c, -1), normalized_features.view(b, c, -1).transpose(1, 2))
    identity_matrix = torch.eye(c, device=features.device).unsqueeze(0).repeat(b, 1, 1)
    loss = torch.mean((inner_product - identity_matrix) ** 2)
    return loss


def correlation_coefficient_loss(feat):
    b, c, h, w = feat.size()
    feat_flat = feat.view(b, c, -1)

    mean_feat = torch.mean(feat_flat, dim=-1, keepdim=True)
    feat_centered = feat_flat - mean_feat

    corr_matrix = torch.bmm(feat_centered, feat_centered.transpose(1, 2)) / (h * w)

    corr_loss = torch.mean(corr_matrix ** 2)

    return corr_loss


def pearson_correlation_loss(feat):
    b, c, h, w = feat.size()
    feat_flat = feat.view(b, c, -1)
    mean_feat = torch.mean(feat_flat, dim=-1, keepdim=True)
    feat_centered = feat_flat - mean_feat

    std_feat = torch.std(feat_centered, dim=-1, keepdim=True)

    corr_matrix = torch.bmm(feat_centered, feat_centered.transpose(1, 2)) / (h * w)
    corr_matrix = corr_matrix / (std_feat @ std_feat.transpose(1, 2) + 1e-12)

    # corr_matrix = corr_matrix - torch.eye(c).to(feat.device)

    corr_loss = torch.mean(corr_matrix ** 2)
    return corr_loss


def mahalanobis_cov_penalty(feat):
    b, c, h, w = feat.size()
    feat_flat = feat.view(b, c, -1)  # Flatten spatial dimensions
    feat_flat = feat_flat - feat_flat.mean(dim=2, keepdim=True)

    cov = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / (h * w)
    cov_loss = torch.norm(cov - torch.eye(c).to(feat.device), p='fro')
    return cov_loss


class DistanceCorrelation:
    def __init__(self, device='cuda'):
        self.device = device

    def _euclidean_distance_matrix(self, x):
        # 计算欧几里得距离矩阵
        x = x.view(x.size(0), -1)  # 将张量展平
        diff = x[:, None, :] - x[None, :, :]  # 计算差异
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-10)  # 防止零距离
        return dist_matrix

    def distance_correlation(self, a, b):
        # 计算距离相关性
        n = a.size(0)

        # 计算距离矩阵
        d_a = self._euclidean_distance_matrix(a)
        d_b = self._euclidean_distance_matrix(b)

        # 计算中心化的距离矩阵
        mean_d_a = d_a.mean(dim=1, keepdim=True)
        mean_d_b = d_b.mean(dim=1, keepdim=True)
        d_a_centered = d_a - mean_d_a - mean_d_a.t() + d_a.mean()
        d_b_centered = d_b - mean_d_b - mean_d_b.t() + d_b.mean()

        # 计算距离相关性
        dcov_ab = (d_a_centered * d_b_centered).sum() / (n ** 2)
        dcov_aa = (d_a_centered * d_a_centered).sum() / (n ** 2)
        dcov_bb = (d_b_centered * d_b_centered).sum() / (n ** 2)

        if dcov_aa == 0 or dcov_bb == 0:
            return torch.tensor(0.0) # 如果某一项为零，则返回零相关性

        # 距离相关性公式
        distance_correlation_value = dcov_ab / torch.sqrt(dcov_aa * dcov_bb)
        return distance_correlation_value


# Training Classifier
def train_fcloud(classifier, device, dataloader, optimizer, epoch, dataset_size):
    # import setting and classifier
    args = parser.parse_args()
    classifier.train()

    if epoch > 300:
        classifier.freeze_until_up()

    correct, total_covariance, total_entropy, total_nonlinear = 0, 0, 0, 0
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            feat, output = classifier(data, block_idx=args.block_idx, release=False)
            smoothed_target = label_smooth(target.long(), args.nz, args.smooth_factor)

            # Calculation of loss
            # loss = F.nll_loss(output, target.long())
            # loss = label_smoothing_cross_entropy_loss(output, smoothed_target)
            loss = F.kl_div(output, smoothed_target, reduction='batchmean')
            # Get the gradient and optimize the classifier
            total_loss = loss

            distance_corr = DistanceCorrelation()
            h_xz_loss = distance_corr.distance_correlation(data, feat[args.block_idx-1])
            total_nonlinear += h_xz_loss.item()
            temp_mutual_information = args.beta * h_xz_loss

            total_loss = args.alpha * total_loss + temp_mutual_information

            entropy = torch.sum(torch.abs(feat[args.block_idx-1])) / feat[args.block_idx-1].numel()
            total_entropy += entropy.item()

            covariance = pearson_correlation_loss(feat[args.block_idx-1])
            total_covariance += covariance.item()

            total_loss = total_loss + 3e-2 * entropy + args.gamma * covariance

            if epoch > 300:
                total_loss = loss
                total_loss.backward()
                optimizer.step()
            else:
                total_loss.backward()
                optimizer.step()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            pbar.set_postfix(zy_loss=loss.item(), h_xz_loss=h_xz_loss.item(), entropy=entropy.item(), covariance=covariance.item(),
                            accuracy=100. * correct / dataset_size)
            # pbar.set_postfix(zy_loss=loss.item(),
            #                  accuracy=100. * correct / dataset_size)
            pbar.update()

            # pbar.set_postfix(zy_loss=loss.item(), h_xz_shuffle_loss=(args.alpha * h_xz_shuffle_loss).item(),
            #                  h_lsy_z_loss=(args.beta * h_lsy_z_loss).item(), accuracy=100. * correct / dataset_size)
            # pbar.update()

    # Output consumed time and training set accuracy after finishing an epoch round
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, Learning Rate: {current_lr}')
    print('Accuracy: {}/{} ({:.4f}%)'.
          format(correct, dataset_size, 100. * correct / dataset_size))
    print('zy_loss: {:.5f}'.format(loss.item()))
    print('avg_nonlinear: {:.5f}, avg_covariance: {:.5f}, avg_entropy: {:.5f}'.format(total_nonlinear / len(dataloader),
          total_covariance / len(dataloader), total_entropy / len(dataloader)))


def test_fcloud(classifier, device, data_loader):
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
            smoothed_target = label_smooth(target.long(), args.nz, args.smooth_factor)
            feat, output = classifier(data, block_idx=args.block_idx, release=False)

            # test_loss += F.kl_div(output, smoothed_target, reduction='sum').item()
            # test_loss += label_smoothing_cross_entropy_loss(output, smoothed_target).item()
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()
            # test_loss += F.cross_entropy(output, smoothed_target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    # test_loss /= len(data_loader)
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset), test_loss


def main():
    # Initialize the log folder and export the records of this training
    args = parser.parse_args()
    os.makedirs(f'log/project_InforDR/{args.dataset}', exist_ok=True)
    sys.stdout = Logger(f'./log/project_InforDR/{args.dataset}/{args.dataset}_{args.target_model}_{args.lr}_{args.block_idx}_pv4_kci_{args.smooth_factor}_train_log.txt')
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
        classifier = pv4_classifier_4(nc=args.nc, ndf=args.ndf, nz=args.nz, img_size=args.img_size, block_idx=args.block_idx).to(device)

        # # Translated comment，允许 block_idx 作为参数传入
        # class WrappedModel(torch.nn.Module):
        #     def __init__(self, model, block_idx):
        #         super(WrappedModel, self).__init__()
        #         self.model = model
        #         self.block_idx = block_idx
        #
        #     def forward(self, x):
        #         return self.model(x, block_idx=self.block_idx)
        #
        # wrapped_model = WrappedModel(classifier, block_idx=args.block_idx)

        summary(classifier, (args.nc, args.img_size, args.img_size))
        with torch.no_grad():
            flops, params = summary_info(classifier, (args.nc, args.img_size, args.img_size), as_strings=True)
            print(f"FLOPs: {flops}, Params: {params}")
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True, weight_decay=1e-3)
            # optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, amsgrad=True, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25,
                                                                   verbose=True)

    elif args.target_model == "resnet":
        classifier = rn18_pv4(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to(device)
        summary(classifier, (args.nc, args.img_size, args.img_size))
        with torch.no_grad():
            flops, params = summary_info(classifier, (args.nc, args.img_size, args.img_size), as_strings=True)
            print(f"FLOPs: {flops}, Params: {params}")
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
                                                                   verbose=True)
    elif args.target_model == 'vgg':
        classifier = vgg16_bn_pv4(nc=args.nc, nz=args.nz, img_size=args.img_size).to(device)
        summary(classifier, (args.nc, args.img_size, args.img_size))
        if args.optimize == "sgd":
            optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.8, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.52, patience=10,
                                                                   verbose=True)
        elif args.optimize == "adam":
            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True,
                                   weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=19,
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
    best_loss_vae = 999
    dataset_size = len(train_loader.dataset)

    base_dir = f'/home/{args.user}/desktop/liurk/CORECODE/CODE/target_model'
    path = get_pth_paths(base_dir, args.dataset, args.target_model)
    print(path)

    checkpoint = torch.load(path, weights_only=True)
    best_acc = checkpoint['best_cl_acc']
    epoch = checkpoint['epoch']
    print("=> loaded unprotect classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_acc))

    # Train classifier
    loss_lst, cl_acc_lst = [], []
    adjusted = False
    for epoch in range(1, args.epochs + 1):

        if epoch > 300 and not adjusted:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
                param_group['betas'] = (0.5, param_group['betas'][0])
                param_group['weight_decay'] = 0.001
            adjusted = True

        train_fcloud(classifier, device, train_loader, optimizer, epoch, dataset_size)
        cl_acc, loss = test_fcloud(classifier, device, test_loader)

        loss_lst.append(loss)
        cl_acc_lst.append(cl_acc)

        if args.scheduler == "yes":
            scheduler.step(loss)

        if epoch > 100:
            if cl_acc > best_cl_acc:
                best_cl_acc = cl_acc
                best_cl_epoch = epoch
                state = {
                    'epoch': epoch,
                    'model': classifier.state_dict(),
                    'best_cl_acc': best_cl_acc,
                }
                os.makedirs(f'./info_protect_target_model/{args.dataset}', exist_ok=True)
                torch.save(state, f'./info_protect_target_model/{args.dataset}/{args.dataset}_{args.target_model}_{args.block_idx}_pv4_{args.smooth_factor}_{args.alpha}_{args.beta}_{args.gamma}_{args.lr}.pth')

    print("Best classifier: epoch {}, acc {:.4f}, drop acc {:.4f}".format(best_cl_epoch, best_cl_acc, best_acc-best_cl_acc))
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
    plt.savefig(f'./info_protect_target_model/{args.dataset}_{args.target_model}_{args.lr}.png')
    plt.show()


# Initiation!
if __name__ == '__main__':
    main()
