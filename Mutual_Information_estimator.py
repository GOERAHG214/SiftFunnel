from __future__ import print_function
import argparse
import random
import sys

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import torch.optim as optim
import re

from torchvision import transforms

from InfoDR_package import pv4_classifier_4, rn18_pv4, vgg16_bn_pv4
from T_model_packages import *
from dpsgd_model_packages import Classifier_4_dpsgd, rn18_dpsgd
from mid_model_packages import Classifier_4_mid, rn18_mid
from target_model_packages import *
from tqdm import tqdm
from utilis import *
from datetime import datetime


current_time = datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch_size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--allocation', type=float, default=0.8, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--scale_factor', type=int, default=8)
parser.add_argument('--block_idx', type=int, default=1)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--nz', type=int, default=2)
parser.add_argument('--truncation', type=int, default=2)
parser.add_argument('--sigma2', type=int, default=24)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--r', type=float, default=0.8)
parser.add_argument('--delta', type=float, default=0.8)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--dataset', default='mnist', metavar='')
parser.add_argument('--user', default='lenovo')
parser.add_argument('--target_model', default='cnn', help='cnn | vgg | ir152 | resnet')
parser.add_argument('--method', default='mas', help='inversion | amplify')
parser.add_argument('--optimize', default='adam', help='sgd | adam')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--set_type', default='sub', help='sub | cross')
parser.add_argument('--protect', default='normal', help='normal | dp | drop')
parser.add_argument('--scheduler', default='yes', help='yes | no')


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
    # Translated comment
    pattern = re.compile(rf'{re.escape(dataset)}_(?P<target_model>.+?)\.pth')

    # Translated comment
    for root, _, files in os.walk(base_dir):
        for file in files:
            # Translated comment .pth 文件
            if file.endswith(".pth"):
                match = pattern.match(file)
                if match:
                    file_target_model = match.group('target_model')
                    # Translated comment
                    if file_target_model.startswith(target_model):
                        # Translated comment
                        full_path = os.path.join(root, file)
                        return full_path


def mine_loss(mine_t, x, y):
    batch_size = x.size(0)

    t_xy = mine_t(x, y).mean()

    y_shuffle = y[torch.randperm(batch_size)]
    t_x_y_shuffle = mine_t(x, y_shuffle)

    t_x_y_shuffle = torch.log(torch.mean(torch.exp(t_x_y_shuffle)))

    loss = t_x_y_shuffle - t_xy
    return loss


def train(classifier, mine, log_interval, device, data_loader, optimizer, epoch):
    args = parser.parse_args()
    classifier.eval()
    mine.train()

    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                              ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

    start_time = time.time()

    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                if args.protect == 'normal':
                    feat, prediction = classifier(transform_amplify(data), release=True)
                    # feat, mu, std, prediction = classifier(transform_amplify(data), release=True)
                    feat_out = feat[args.block_idx-1]

                elif args.protect == 'dp':
                    feat, prediction = classifier(transform_amplify(data), release=True)
                    gaussian_noise = torch.randn_like(feat[args.block_idx - 1]) * args.delta
                    feat_out = feat[args.block_idx - 1] + gaussian_noise
                elif args.protect == 'drop':
                    feat, prediction = classifier(transform_amplify(data), release=True)
                    mask = torch.bernoulli(torch.full_like(feat[args.block_idx - 1], 1 - args.r))
                    feat_out = feat[args.block_idx - 1] * mask

            loss = mine_loss(mine, transform_amplify(data), feat_out)
            # loss = mine.learning_loss(transform_amplify(data).view(data.size(0), -1), feat_out.view(data.size(0), -1))
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(mine.parameters(), max_norm=1.0)
            optimizer.step()

            elapsed_time = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

            pbar.set_postfix(loss=loss.item(), time=elapsed_str)
            pbar.update()

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, Learning Rate: {current_lr}')


def test(classifier, mine, device, data_loader):
    args = parser.parse_args()
    classifier.eval()
    mine.eval()

    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                                ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

    loss, non_zero_count, data_cnt, cnt = 0, 0, 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if args.protect == 'normal':
                feat, prediction = classifier(transform_amplify(data), release=True)
                # feat, mu, std, prediction = classifier(transform_amplify(data), release=True)
                feat_out = feat[args.block_idx - 1]

            elif args.protect == 'dp':
                feat, prediction = classifier(transform_amplify(data), release=True)
                gaussian_noise = torch.randn_like(feat[args.block_idx - 1]) * args.delta
                feat_out = feat[args.block_idx - 1] + gaussian_noise
            elif args.protect == 'drop':
                feat, prediction = classifier(transform_amplify(data), release=True)
                mask = torch.bernoulli(torch.full_like(feat[args.block_idx - 1], 1 - args.r))
                feat_out = feat[args.block_idx - 1] * mask

            loss += -(mine_loss(mine, transform_amplify(data), feat_out)).item()
            # loss += mine.learning_loss(transform_amplify(data).view(data.size(0), -1), feat_out.view(data.size(0), -1)).item()
            non_zero_count += torch.count_nonzero(feat_out)
            data_cnt += data.size(0)
            cnt += 1

    avg_loss = loss / cnt
    non_zero_count = non_zero_count / data_cnt
    print("\nThe average mutual information estimate between block {} input and features is: {:.5f}. delta z is {}"
          .format(args.block_idx, avg_loss, non_zero_count))
    return  avg_loss


def evaluate_inversion_model(inversion, classifier, test_loader, device):
    args = parser.parse_args()
    classifier.eval()
    inversion.eval()

    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                                ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])
    non_zero_count, data_cnt = 0, 0

    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            images = images.to(device)

            if args.protect == 'normal':
                feat, prediction = classifier(transform_amplify(images), release=True)
                # feat, mu, std, prediction = classifier(transform_amplify(data), release=True)
                feat_out = feat[args.block_idx - 1]

            elif args.protect == 'dp':
                feat, prediction = classifier(transform_amplify(images), release=True)
                gaussian_noise = torch.randn_like(feat[args.block_idx - 1]) * args.delta
                feat_out = feat[args.block_idx - 1] + gaussian_noise
            elif args.protect == 'drop':
                feat, prediction = classifier(transform_amplify(images), release=True)
                mask = torch.bernoulli(torch.full_like(feat[args.block_idx - 1], 1 - args.r))
                feat_out = feat[args.block_idx - 1] * mask

            non_zero_count += torch.count_nonzero(feat_out)
            data_cnt += images.size(0)

    non_zero_count = non_zero_count / data_cnt

    return non_zero_count


def main():
    # Initialize the log folder and export the records of this training
    args = parser.parse_args()
    os.makedirs(f'log/mutual_information/{args.dataset}', exist_ok=True)
    sys.stdout = Logger(f'./log/mutual_information/{args.dataset}/{args.dataset}_{args.target_model}_{args.block_idx}_train_log_bido.txt')
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
    transform = transforms.Compose([transforms.ToTensor()])

    # The definition of target model
    if args.target_model == "cnn":
        classifier = AE_Classifier_4(nc=args.nc, ndf=args.ndf, nz=args.nz, img_size=args.img_size).to(device)
        # Define inversion model
        output_channels_dict = {
            1: args.ngf,  # Assuming the output channels of block 1 are ngf
            2: args.ngf,  # Assuming the output channels of block 2 are ngf * 2
            3: args.ngf * 4,  # Assuming the output channels of block 3 are ngf * 4
            4: args.ngf * 8  # Assuming the output channels of block 4 are ngf * 8
        }
        output_channels = output_channels_dict.get(args.block_idx,
                                                 args.ngf * 2)  # Default to args.ngf if block_idx is not found
        tnet = ConvMINE(args.nc, output_channels, args.img_size).to(device)
        # tnet = CLUB(args.nc * args.img_size * args.img_size, output_channels * 16 * 16, args.img_size).to(device)
        optimizer = optim.AdamW(tnet.parameters(), lr=args.lr, betas=(0.8, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=True)

    elif args.target_model == "resnet":
        classifier = rn18_pv4(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to(device)
        input_channels_dict = {
            1: 2,  # Assuming the output channels of block 1 are ngf
            2: args.ngf * 2,  # Assuming the output channels of block 2 are ngf * 2
            3: args.ngf,  # Assuming the output channels of block 3 are ngf * 4
            4: args.ngf * 2,  # Assuming the output channels of block 4 are ngf * 8
            5: args.ngf * 2
        }
        # Get the output size and input channels for the current block_idx
        output_channels = 64
        if args.block_idx <= 3:
            output_channels = input_channels_dict.get(1, 2)  # Default to args.ngf if block_idx is not found

        if args.block_idx in range(4, 6):
            output_channels = input_channels_dict.get(2, args.ngf)  # Default to args.ngf if block_idx is not found

        if args.block_idx in range(6, 8):
            output_channels = input_channels_dict.get(3, args.ngf)  # Default to args.ngf if block_idx is not found

        if args.block_idx in range(8, 11):
            output_channels = input_channels_dict.get(4, args.ngf)  # Default to args.ngf if block_idx is not found

        tnet = ConvMINE(args.nc, output_channels, args.img_size).to(device)
        # tnet = CLUB(args.nc * args.img_size * args.img_size, output_channels * 64 * 64, args.img_size).to(device)
        optimizer = optim.AdamW(tnet.parameters(), lr=args.lr, betas=(0.6, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=True)

    elif args.target_model == 'vgg':
        classifier = vgg16_bn_pv4(nc=args.nc, nz=args.nz, img_size=args.img_size).to(device)
        output_channels_dict = {
            1: args.ngf,  # Assuming the output channels of block 1 are ngf
            2: 8,  # Assuming the output channels of block 2 are ngf * 2
            3: 128,  # Assuming the output channels of block 3 are ngf * 4
            4: args.ngf * 8  # Assuming the output channels of block 4 are ngf * 8
        }
        output_channels = output_channels_dict.get(2,
                                                   args.ngf * 2)  # Default to args.ngf if block_idx is not found
        tnet = ConvMINE(args.nc, output_channels, args.img_size).to(device)
        # tnet = CLUB(args.nc * args.img_size * args.img_size, output_channels * 16 * 16, args.img_size).to(device)
        optimizer = optim.AdamW(tnet.parameters(), lr=args.lr, betas=(0.8, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=True)

    elif args.target_model == "ir152":
        classifier = IR152(args.nz, args.nc, args.img_size).to(device)

    # The definition of data set
    train_set, test_set = None, None
    if args.dataset == 'facescrub':
        train_set = CelebA(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform)
        # Inversion attack on TRAIN data of facescrub classifier
        test1_set = FaceScrub(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=True)
        # Inversion attack on TEST data of facescrub classifier
        test2_set = FaceScrub(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=False)

    elif args.dataset == 'chest':
        train_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, attack=True)
        # Inversion attack on TRAIN data of chest classifier
        test1_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=True)
        # Inversion attack on TEST data of chest classifier
        test2_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=False)

    elif args.dataset == 'cifar10':
        train_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', attack=True,
                               transform=transform, )
        # Inversion attack on TRAIN data of cifar64 classifier
        test1_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', train=True, transform=transform)
        # Inversion attack on TEST data of cifar64 classifier
        test2_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Load classifier
    os.makedirs(f'mutual_information_estimator/{args.dataset}/{args.target_model}', exist_ok=True)

    base_dir = f'/home/{args.user}/desktop/liurk/CORECODE/CODE/target_model'
    # path = get_pth_paths(base_dir, args.dataset, args.target_model)
    # path = "/home/yons/desktop/liurk/CORECODE/CODE/info_protect_target_model/cifar10/cifar10_resnet_2_pv4_kci_0.35.pth"
    print(path)

    checkpoint = torch.load(path, weights_only=True)

    state_dict = checkpoint['model']
    state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
    classifier.load_state_dict(state_dict)

    # classifier.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))

    best_mine_loss = 0
    recon_mine_test_list = []
    for epoch in range(1, args.epochs + 1):
        train(classifier, tnet, args.log_interval, device, test1_loader, optimizer, epoch)
        recon_loss_test = test(classifier, tnet, device, train_loader)
        loss_t = -recon_loss_test

        if args.scheduler == "yes":
            scheduler.step(-loss_t)

        recon_mine_test_list.append(recon_loss_test)

        if recon_loss_test > best_mine_loss:
            best_mine_loss = recon_loss_test
            state = {
                'epoch': epoch,
                'model': tnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_mine_loss
            }
            torch.save(state, f'mutual_information_estimator/{args.dataset}/{args.target_model}/inversion_{args.block_idx}_info.pth')

    non = evaluate_inversion_model(tnet, classifier, test2_loader, device)
    print("delta z is: {:.5f}".format(non))
    print("Best best_mine_loss is {:.5f}".format(best_mine_loss))

    # Plotting recon loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), recon_mine_test_list, label='MINE')
    plt.xlabel('Epoch')
    plt.ylabel('Mutual Information Estimation')
    plt.title('MINE')
    plt.legend()
    plt.savefig(f'mutual_information_estimator/{args.dataset}/{args.target_model}/inversion_{args.block_idx}.jpg')
    plt.close()
    print("Formatted time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))


# Initiation!
if __name__ == '__main__':
    main()