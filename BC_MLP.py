import torch.nn.functional as F
import torch.nn as nn
import torch, os, random, time, sys, argparse, datetime, pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from timm.utils import AverageMeter
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score
from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet50
from thop import profile

from dataset.OC_dataset import GatherDataset
from model import BC_MLP
from logger import Logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def test(args):
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # define the model
    model = BC_MLP(device=DEVICE, resume=args.resume).to(DEVICE)
    model.eval()

    if args.only_gan:
        fake_items = [
            'progan', 'stargan', 'pigan',
            'stylegan2', 'vqgan',
        ]
    elif args.only_diffusion:
        fake_items = [
            'ldm', 'ddim', 'sdv21',
            'freeDom', 'hps', 'midj', 'sdxl'
        ]
    elif args.only_paper:
        fake_items = [
            'stylegan2', 'vqgan',
            'ldm', 'ddim', 'sdv21',
            'freeDom', 'hps', 'midj', 'sdxl'
        ]
    else:
        fake_items = [
        'progan', 'stargan', 'pigan',
        'stylegan2', 'vqgan',
        'ldm', 'ddim', 'sdv21',
        'freeDom', 'hps', 'midj', 'sdxl'
    ]

    AUC_meter = []
    AP_meter = []
    Acc_meter = []
    for dataset_item in fake_items:
        dataset_real = GatherDataset(
            real_item_tag='celeba', only_real=True, only_fake=False,
            fake_items_tag=dataset_item,
            training=False,
            random_subset=True,
            random_subset_num_fake=5000,
            random_subset_num_real=5000,
        )
        print(f'length of {dataset_item} - photographic: {len(dataset_real)}')
        dataloader_real = torch.utils.data.DataLoader(
            dataset_real,
            batch_size=256,
            shuffle=True,
            num_workers=4
        )

        dataset_fake = GatherDataset(
            only_fake=True, only_real=False,
            fake_items_tag=dataset_item, training=False,
            random_subset=True,
            random_subset_num_fake=5000,
            random_subset_num_real=5000,
        )
        dataloader_fake = torch.utils.data.DataLoader(
            dataset_fake,
            batch_size=256,
            shuffle=True,
            num_workers=4
        )
        print(f'length of {dataset_item} - generated: {len(dataset_fake)}')

        y_true, y_pred = [], []
        with torch.no_grad():
            for img in tqdm(dataloader_real):
                in_tens = img['img'].cuda()
                label = [0] * len(in_tens)
                outputs = model(in_tens)

                y_pred.extend(outputs.softmax(dim=1)[:, 1].flatten().tolist())
                y_true.extend(label)

            for img in tqdm(dataloader_fake):
                in_tens = img['img'].cuda()
                label = [1] * len(in_tens)

                outputs = model(in_tens)

                y_pred.extend(outputs.softmax(dim=1)[:, 1].flatten().tolist())
                y_true.extend(label)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        print("({}) acc: {:.2f}; auc {:.2f}; ap: {:.2f}".format(dataset_item, acc * 100, auc * 100, ap * 100))
        AUC_meter.append(auc)
        AP_meter.append(ap)
        Acc_meter.append(acc)

    # calculate mean number
    print(f"Mean number of Acc: {np.mean(Acc_meter): .4f}")
    print(f"Mean number of AP: {np.mean(AP_meter): .4f}")
    print(f"Mean number of AUC: {np.mean(AUC_meter): .4f}")

    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # for FLOPS
    # 2. inference time
    model.eval()  
    with torch.no_grad():
        start_time = time.time()
        model(dummy_input)  # forward
        end_time = time.time()

    inference_time = end_time - start_time  
    print(f"Inference Time (per forward pass): {inference_time:.6f} seconds")

    # 3. FLOPS
    flops_per_second = flops / inference_time  
    flops_per_second_g = flops_per_second / 1e9  #  GFLOPS

    print(f"FLOPS (Floating Point Operations Per Second): {flops_per_second_g:.2f} GFLOPS")



if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse.add_argument('--seed', type=int, default=2024)
    parse.add_argument('--resume', type=str, default='./pretrained/BC.pth')
    parse.add_argument('--output', type=str, default='./output/test/')
    parse.add_argument('--name', type=str, default='BC_MLP')

    # for testing choice
    parse.add_argument('--only_gan', action='store_true')
    parse.add_argument('--only_diffusion', action='store_true')
    parse.add_argument('--only_paper', action='store_true')
    parse.add_argument('--complexity_cal', action='store_true')
    args = parse.parse_args()

    output_path = os.path.join(args.output, args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    Logger(os.path.join(output_path, 'log.log'))
    print('  '.join(list(sys.argv)))


    test(args)
