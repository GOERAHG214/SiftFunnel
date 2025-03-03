from __future__ import print_function
import argparse
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.optim as optim
from tqdm import tqdm

from dpsgd_model_packages import Classifier_4_dpsgd, rn18_dpsgd
from mid_model_packages import Classifier_4_mid, rn18_mid
from target_model_packages import *
from utilis import *
from InfoDR_package import *
import torchvision.utils as vutils
from datetime import datetime
import re


current_time = datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch_size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='')
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
parser.add_argument('--Niters', type=int, default=500)
parser.add_argument('--truncation', type=int, default=2)
parser.add_argument('--sigma2', type=int, default=24)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--lambda_TV', type=float, default=1e3)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--dataset', default='mnist', metavar='')
parser.add_argument('--user', default='lenovo', metavar='')
parser.add_argument('--target_model', default='cnn', help='cnn | vgg | ir152 | resnet')
parser.add_argument('--method', default='mas', help='inversion | amplify')
parser.add_argument('--optimize', default='adam', help='sgd | adam')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--set_type', default='sub', help='sub | cross')
parser.add_argument('--scheduler', default='no', help='yes | no')

torch.autograd.set_detect_anomaly(True)

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


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()


def cosine_similarity(tensor1, tensor2):
    cos_sim = F.cosine_similarity(tensor1.view(tensor1.size(0), -1), tensor2.view(tensor2.size(0), -1), dim=1)
    return cos_sim.mean().item()


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = ((x[:,:,1:,:]-x[:,:,:h_x-1,:]) ** 2).sum()
    w_tv = ((x[:,:,:,1:]-x[:,:,:,:w_x-1]) ** 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x**2).mean()


def normalize_0_1(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def Process(im_flatten):
    maxValue = torch.max(im_flatten)
    minValue = torch.min(im_flatten)
    im_flatten = im_flatten - minValue
    im_flatten = im_flatten / (maxValue - minValue)
    return im_flatten


def test(classifier, device, data_loader, msg):
    args = parser.parse_args()
    classifier.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                                ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

    mse_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_cosine_similarity = 0
    count, epoch_cnt, data_len = 0, 0, 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        feat, prediction = classifier(transform_amplify(data), release=True)
        # feat, _, _, _ = classifier(transform_amplify(data), release=True)

        reconstruction = torch.randn(data.size(), requires_grad=True, device="cuda")
        optimizer = optim.SGD(params=[reconstruction], lr=args.lr, momentum=0.9)

        start_time = time.time()
        min_loss, cnt, lr = 9999, 0, args.lr
        with tqdm(total=args.Niters, desc=f"Computing {epoch_cnt}") as pbar:
            for i in range(args.Niters):
                if reconstruction.grad is not None:
                    reconstruction.grad.zero_()
                reconstruction_feat, _ = classifier(transform_amplify(reconstruction), release=True)
                # reconstruction_feat, _, _, _ = classifier(transform_amplify(reconstruction), release=True)
                feat_loss = ((reconstruction_feat[args.block_idx - 1] - feat[args.block_idx - 1])**2).mean()
                tv_loss = TV(reconstruction)
                normal_loss = l2loss(reconstruction)

                total_loss = feat_loss + args.lambda_TV * (tv_loss ** 0.5)
                total_loss.backward(retain_graph=True)
                optimizer.step()

                # Translated comment
                elapsed_time = time.time() - start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                pbar.set_postfix(feat_loss=feat_loss.item(), total_loss=total_loss.item(), time=elapsed_str)
                pbar.update()

        # reconstruction = normalize_0_1(reconstruction)
        reconstruction = torch.clamp(reconstruction, 0, 1)

        mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

        psnr_value = psnr(reconstruction, data, max_val=1.0)
        total_psnr += psnr_value

        if args.nc == 1:
            ssim_value = ssim(reconstruction.cpu().detach().numpy(), data.cpu().numpy(), data_range=1.0, channel_axis=1, multichannel=False)
        else:
            ssim_value = ssim(reconstruction.cpu().detach().numpy(), data.cpu().numpy(), data_range=1.0, channel_axis=1, multichannel=True)
        total_ssim += ssim_value

        cosine_sim = cosine_similarity(reconstruction, data)
        total_cosine_similarity += cosine_sim

        data_len += int(data.size(0))

        print("Count {}: MSE: {:5e}, PSNR: {:.6f}, SSIM: {:.6f}"
              .format(epoch_cnt, mse_loss / (data_len * args.img_size * args.img_size), psnr_value, ssim_value))

        count += 1

        truth = data[0:32]
        inverse = reconstruction[0:32]
        out = torch.cat((truth, inverse))

        for i in range(4):
            out[i * 16:i * 16 + 8] = truth[i * 8:i * 8 + 8]
            out[i * 16 + 8:i * 16 + 16] = inverse[i * 8:i * 8 + 8]

        out = out.flip(1)
        # Save image
        os.makedirs(f'out/white_box_attack/{args.dataset}/{args.target_model}/{args.block_idx}', exist_ok=True)
        vutils.save_image(out, 'out/white_box_attack/{}/{}/{}/recon_{}_{}.png'.format(args.dataset, args.target_model,
                                                                       args.block_idx, msg.replace(" ", ""), epoch_cnt),
                          normalize=False)
        epoch_cnt += 1

    mse_loss /= len(data_loader.dataset) * args.img_size * args.img_size
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_cosine_similarity = total_cosine_similarity / count
    print('\nTest inversion model on {} set: Average MSE loss: {:.f}, Average PSNR loss: {:.6f}, '
          'Average SSIM loss: {:.6f},Average Cosine Similarity: {:.6f}\n'.format(msg, mse_loss, avg_psnr,
                                                                                 avg_ssim, avg_cosine_similarity))
    return mse_loss


def main():
    args = parser.parse_args()
    os.makedirs(f'log/white_box_attack/{args.dataset}', exist_ok=True)
    sys.stdout = Logger(f'./log/white_box_attack/{args.dataset}/{args.dataset}_{args.target_model}_{args.block_idx}_{args.lr}_ae_train_log.txt')
    print("Formatted time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("================================")
    print(args)
    print("================================")
    os.makedirs('out', exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor(),])

    if args.dataset == 'facescrub':
        # Inversion attack on TRAIN data of facescrub classifier
        test1_set = FaceScrub(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=True)
        # Inversion attack on TEST data of facescrub classifier
        test2_set = FaceScrub(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=False)

    elif args.dataset == 'chest':
        # Inversion attack on TRAIN data of chest classifier
        test1_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=True)
        # Inversion attack on TEST data of chest classifier
        test2_set = Chest(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', transform=transform, train=False)

    elif args.dataset == 'cifar10':
        # Inversion attack on TRAIN data of cifar64 classifier
        test1_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', train=True, transform=transform)
        # Inversion attack on TEST data of cifar64 classifier
        test2_set = CIFAR10_64(root=f'/home/{args.user}/desktop/liurk/CORECODE/dataset', train=False, transform=transform)

    test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # The definition of target model
    if args.target_model == "cnn":
        classifier = Classifier_4(nc=args.nc, ndf=args.ndf, nz=args.nz, img_size=args.img_size).to(device)

    elif args.target_model == "resnet":
        classifier = rn18(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to(device)

    elif args.target_model == 'vgg':
        classifier = vgg16_bn_pv4(nc=args.nc, nz=args.nz, img_size=args.img_size).to(device)

    elif args.target_model == "ir152":
        classifier = IR152(args.nz, args.nc, args.img_size).to(device)


    # Load classifier
    base_dir = f'/home/{args.user}/desktop/liurk/CORECODE/CODE/target_model'
    # path = get_pth_paths(base_dir, args.dataset, args.target_model)
    path = "/home/yons/desktop/liurk/CORECODE/CODE/info_protect_target_model/cifar10/cifar10_vgg_2_pv4_kci_0.35_addl12.pth"
    print(path)

    checkpoint = torch.load(path, weights_only=True)

    state_dict = checkpoint['model']
    state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
    classifier.load_state_dict(state_dict)

    # classifier.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))


    # Train inversion model
    # test(classifier, device, test1_loader, 'test1')
    test(classifier, device, test2_loader, 'test2')


if __name__ == '__main__':
    main()