#ss_students@dslab:~/Desktop/paper1/MiDSS$ python3 code/train.py --dataset fundus --lb_domain 1 --lb_num 20 --save_name train1_fundus
#nohup python3 code/train.py --dataset prostate_lesion --lb_domain 1 --lb_num 480 --save_name train_1_480 > train_1_480.log 2>&1 &
import argparse
import logging
import os
import random
import shutil
import sys
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from networks.lesion_models import build_segmentation_model
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, ProstateLesionSegmentation, MNMSSegmentation, _find_existing_modality_dir
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'prostate_lesion', 'MNMS'])
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument(
    "--model",
    type=str,
    default="unet",
    choices=["unet", "fpn", "deeplabv3plus", "segformer", "unetpp", "attunet", "resunet", "scseunet"],
    help="model_name"
)
parser.add_argument('--encoder_name', type=str, default=None,
                    help='encoder for SMP models: fpn/deeplabv3plus default resnet34, segformer default mit_b0')
parser.add_argument('--encoder_weights', type=str, default='imagenet',
                    help='encoder weights for SMP models; use None to train encoder from scratch')
parser.add_argument("--max_iterations", type=int, default=None, help="maximum training iterations; if omitted, dataset-specific default is used")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--load',action='store_true')
parser.add_argument('--load_path',type=str,default='../model/lb1_ratio0.2/iter_6000.pth')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument("--LB", default=0.01, type=float)
parser.add_argument('--data_root', type=str, default=None,
                    help='dataset root path; prostate_lesion defaults to ../../data/Prostate_Lesion3')
parser.add_argument('--lesion_modalities', type=str, default='t2w,adc',
                    help='comma-separated registered modality names for prostate_lesion. Default t2w,adc resolves to image_t2w,image_adc layout')
parser.add_argument('--lesion_modality_dirs', type=str, default=None,
                    help='optional comma-separated explicit folder names matching lesion_modalities')
parser.add_argument('--lesion_norm', type=str, default='minmax', choices=['legacy', 'minmax', 'zscore'])
parser.add_argument('--add_adc_sobel', type=int, default=0, help='append ADC Sobel gradient channel for lesion data')
parser.add_argument('--lesion_sampling_prob', type=float, default=0.7,
                    help='target positive-slice sampling probability for labeled lesion batches')
parser.add_argument('--lesion_sample_unlabeled', type=int, default=0,
                    help='also balance unlabeled loader using masks when available')
parser.add_argument('--lesion_positive_crop_prob', type=float, default=1.0,
                    help='probability of lesion-centered crop when a lesion is present; 1.0 preserves lesions in positive slices')
parser.add_argument('--lesion_crop_jitter', type=float, default=0.35,
                    help='jitter as a fraction of crop size around selected lesion pixel')
parser.add_argument('--lesion_crop_strict', type=int, default=1,
                    help='raise an error if a positive-slice crop cannot preserve lesion pixels')
parser.add_argument('--lesion_cp_prob', type=float, default=1.0, help='probability for lesion-aware Copy-Paste mask')
parser.add_argument('--lesion_cp_dilate', type=int, default=10, help='dilation radius for copied lesion context')
parser.add_argument('--lesion_safe_fda', type=int, default=1, help='protect lesion region from FDA/TP-RAM style mixing')
parser.add_argument('--lesion_fda_LB', type=float, default=0.005, help='weak FDA low-frequency window for lesions')
parser.add_argument('--fda_channels', type=str, default='all', choices=['all', 't2w', 'adc', 'none'])
parser.add_argument('--clahe_prob', type=float, default=0.4)
parser.add_argument('--clahe_clip', type=float, default=2.0)
parser.add_argument('--gamma_prob', type=float, default=0.35)
parser.add_argument('--bias_prob', type=float, default=0.30)
parser.add_argument('--blur_prob', type=float, default=0.15)
parser.add_argument('--boundary_loss_weight', type=float, default=0.1)
parser.add_argument('--boundary_radius', type=int, default=2)
parser.add_argument('--lesion_ce_pos_weight', type=float, default=2.0)
parser.add_argument('--lesion_focal_weight', type=float, default=0.0)
parser.add_argument('--lesion_bg_threshold', type=float, default=0.95)
parser.add_argument('--lesion_fg_threshold', type=float, default=0.75)
parser.add_argument('--post_min_area', type=int, default=0, help='validation post-process: remove lesion components smaller than this')
parser.add_argument('--post_topk', type=int, default=0, help='validation post-process: keep only largest k lesion components when >0')
parser.add_argument('--post_fill_holes', type=int, default=0)

# DIP-oriented lesion domain-generalization block.
# CFET = Contrast/Frequency/Edge/Texture. It is intentionally lightweight and
# plug-in: intensity non-linearity + bias/noise simulation + edge-weighted loss
# and consistency, so it fits a DIP project better than adding a heavy module.
parser.add_argument('--dip_cfet_block', type=int, default=1,
                    help='enable DIP CFET block for prostate_lesion')
parser.add_argument('--dip_aug_prob', type=float, default=0.70,
                    help='probability of tensor-level CFET augmentation')
parser.add_argument('--dip_gamma_min', type=float, default=0.65)
parser.add_argument('--dip_gamma_max', type=float, default=1.55)
parser.add_argument('--dip_contrast', type=float, default=0.25,
                    help='random contrast range around 1.0 for CFET augmentation')
parser.add_argument('--dip_bias_strength', type=float, default=0.20,
                    help='low-frequency multiplicative bias field strength')
parser.add_argument('--dip_noise_std', type=float, default=0.015,
                    help='small acquisition-noise simulation in [0,1] image space')
parser.add_argument('--dip_edge_weight', type=float, default=0.50,
                    help='Sobel edge emphasis added to CE/consistency losses')
parser.add_argument('--dip_sup_weight', type=float, default=0.35,
                    help='extra supervised loss on CFET-augmented labelled images')
parser.add_argument('--dip_cons_weight', type=float, default=0.03,
                    help='prediction consistency between normal and CFET-augmented unlabeled images')
parser.add_argument('--lesion_fda_min_degree', type=float, default=0.20,
                    help='minimum FDA amplitude-mixing degree for lesion training; prevents FDA being almost off in early epochs')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np

def low_freq_mutate_np( amp_src, amp_trg, L=0.1, degree=1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # ratio = random.randint(1,10)/10
    ratio = random.uniform(0, degree)

    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * (1-ratio) + a_trg[:,h1:h2,w1:w2] * ratio
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1, degree=1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img #.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L, degree=degree)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg #.transpose(1, 2, 0)


def split_csv(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]


def tensor_to_255_np(tensor_chw):
    arr = tensor_chw.detach().cpu().numpy().astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        ch = arr[c]
        if ch.min() >= -1.05 and ch.max() <= 1.05:
            out[c] = (ch + 1.0) * 127.5
        else:
            mn, mx = float(ch.min()), float(ch.max())
            out[c] = (ch - mn) / (mx - mn + 1e-6) * 255.0
    return np.clip(out, 0, 255)


def select_fda_channel_indices(modalities, total_channels, mode):
    base_channels = min(len(modalities), total_channels)
    if mode == 'none':
        return []
    if mode == 'all':
        return list(range(base_channels))
    selected = []
    for idx, modality in enumerate(modalities[:base_channels]):
        name = modality.lower()
        if mode == 'adc' and 'adc' in name:
            selected.append(idx)
        if mode == 't2w' and ('t2' in name or name in ['image', 'img', 'image_t2w']):
            selected.append(idx)
    if not selected:
        raise ValueError(
            'Requested --fda_channels={} but no matching channel was found in --lesion_modalities={}. '
            'No fallback to other channels is performed.'.format(mode, modalities)
        )
    return selected


def make_lesion_sampler(dataset_obj, positive_prob):
    presence = getattr(dataset_obj, 'lesion_presence', None)
    if presence is None or len(presence) == 0:
        return None
    presence = np.asarray(presence).astype(np.float32)
    pos = presence == 1
    neg = presence == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    positive_prob = float(np.clip(positive_prob, 0.01, 0.99))
    weights = np.zeros_like(presence, dtype=np.float32)
    weights[pos] = positive_prob / pos.sum()
    weights[neg] = (1.0 - positive_prob) / neg.sum()
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def dilate_binary_tensor(mask, radius):
    if radius <= 0:
        return mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    kernel = radius * 2 + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)


def obtain_lesion_context_mask(label, p=1.0, dilation=10, fallback_random=False):
    boxes = []
    img_size = label.shape[-1]
    for i in range(label.shape[0]):
        cur = label[i].float()
        if random.random() > p:
            boxes.append(torch.zeros_like(cur))
            continue
        if cur.sum() < 1:
            if fallback_random:
                boxes.append(obtain_cutmix_box(img_size=img_size, p=1.0).to(label.device))
            else:
                boxes.append(torch.zeros_like(cur))
            continue
        context = dilate_binary_tensor(cur.unsqueeze(0), dilation).squeeze(0).squeeze(0)
        boxes.append(context.clamp(0, 1))
    return torch.stack(boxes, dim=0).to(label.device)


def protect_frequency_image(freq_img, original_img, label, radius):
    protect = dilate_binary_tensor(label, radius)
    if protect.dim() == 3:
        protect = protect.unsqueeze(1)
    return freq_img * (1.0 - protect) + original_img * protect


def _nhw(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor


def weighted_map_mean(loss_map, mask=None, weight_map=None, eps=1e-6):
    """Mean for pixel losses with optional confidence and edge weights."""
    loss_map = _nhw(loss_map)
    weights = torch.ones_like(loss_map, dtype=loss_map.dtype, device=loss_map.device)
    if mask is not None:
        weights = weights * _nhw(mask).to(device=loss_map.device, dtype=loss_map.dtype)
    if weight_map is not None:
        weights = weights * _nhw(weight_map).to(device=loss_map.device, dtype=loss_map.dtype)
    return (loss_map * weights).sum() / weights.sum().clamp_min(eps)


def weighted_ce_loss(ce_loss_fn, logits, target, mask=None, weight_map=None):
    ce_map = ce_loss_fn(logits, target)
    return weighted_map_mean(ce_map, mask=mask, weight_map=weight_map)


def sobel_edge_weight(img, strength=0.5, eps=1e-6):
    """Return 1 + strength * normalized Sobel magnitude for edge-aware lesion loss."""
    if strength <= 0 or img.dim() != 4:
        return None
    gray = img.mean(dim=1, keepdim=True).float()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + eps)
    max_per_img = grad.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    grad = grad / (max_per_img + eps)
    return 1.0 + float(strength) * grad


def dip_cfet_augment(x, aug_prob=0.7, gamma_min=0.65, gamma_max=1.55,
                     contrast=0.25, bias_strength=0.20, noise_std=0.015):
    """
    DIP CFET augmentation in tensor space.

    It simulates cross-device MRI appearance changes using classical image
    processing operations: nonlinear intensity remapping, contrast scaling,
    low-frequency multiplicative bias field, and mild acquisition noise.
    Input/output are expected in [-1, 1].
    """
    if aug_prob <= 0 or x.dim() != 4:
        return x
    n, c, h, w = x.shape
    apply = (torch.rand(n, 1, 1, 1, device=x.device) < aug_prob).to(dtype=x.dtype)
    x01 = ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    gamma = torch.empty(n, 1, 1, 1, device=x.device).uniform_(gamma_min, gamma_max)
    out = torch.pow(x01.clamp_min(1e-5), gamma)

    if contrast > 0:
        cfac = torch.empty(n, 1, 1, 1, device=x.device).uniform_(1.0 - contrast, 1.0 + contrast)
        mean = out.mean(dim=(2, 3), keepdim=True)
        out = (out - mean) * cfac + mean

    if bias_strength > 0:
        low = torch.empty(n, 1, 4, 4, device=x.device).uniform_(1.0 - bias_strength, 1.0 + bias_strength)
        bias = F.interpolate(low, size=(h, w), mode='bicubic', align_corners=False)
        bias = bias / bias.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        out = out * bias

    if noise_std > 0:
        out = out + torch.randn_like(out) * float(noise_std)

    out = out.clamp(0.0, 1.0) * 2.0 - 1.0
    out = out.to(dtype=x.dtype)
    return x * (1.0 - apply) + out * apply


def pseudo_label_and_mask_from_probs(prob_map, threshold, dataset_name):
    if dataset_name == 'prostate_lesion' and prob_map.shape[1] == 2:
        fg_prob = prob_map[:, 1]
        bg_prob = prob_map[:, 0]
        pseudo = torch.argmax(prob_map, dim=1)
        bg_thr = getattr(args, 'lesion_bg_threshold', threshold)
        fg_thr = getattr(args, 'lesion_fg_threshold', threshold)
        conf_mask = torch.where(pseudo == 1, fg_prob > fg_thr, bg_prob > bg_thr)
        return pseudo, conf_mask.unsqueeze(1).float()
    conf, pseudo = torch.max(prob_map, dim=1)
    return pseudo, (conf > threshold).unsqueeze(1).float()


def get_train_domain_lengths(dataset_name, base_dir, domain_num, primary_modality=None, primary_dir=None):
    if dataset_name == 'fundus':
        domain_name = {1:'Domain1', 2:'Domain2', 3:'Domain3', 4:'Domain4'}
    elif dataset_name == 'prostate':
        domain_name = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}
    elif dataset_name == 'prostate_lesion':
        domain_name = {1:'Dom1', 2:'Dom2', 3:'Dom3'}
    elif dataset_name == 'MNMS':
        domain_name = {1:'vendorA', 2:'vendorB', 3:'vendorC', 4:'vendorD'}
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    lengths = []
    for i in range(1, domain_num + 1):
        if dataset_name == 'prostate_lesion':
            domain_train_dir = os.path.join(base_dir, domain_name[i], 'train')
            resolved = _find_existing_modality_dir(domain_train_dir, primary_modality or 't2w', primary_dir)
            image_dir = os.path.join(domain_train_dir, resolved)
        else:
            image_dir = os.path.join(base_dir, domain_name[i], 'train', primary_dir or 'image')
        files = sorted(glob(os.path.join(image_dir, '*.png')))
        if len(files) == 0:
            raise FileNotFoundError('No PNG files found while counting training images: {}'.format(image_dir))
        lengths.append(len(files))
    return lengths


if args.dataset == 'fundus':
    part = ['cup', 'disc']
    dataset = FundusSegmentation
elif args.dataset == 'prostate':
    part = ['base']
    dataset = ProstateSegmentation
elif args.dataset == 'prostate_lesion':
    part = ['lesion']
    dataset = ProstateLesionSegmentation
elif args.dataset == 'MNMS':
    part = ['lv', 'myo', 'rv']
    dataset = MNMSSegmentation
n_part = len(part)
dice_calcu = {
    'fundus': metrics.dice_coeff_2label,
    'prostate': metrics.dice_coeff,
    'prostate_lesion': metrics.dice_coeff,
    'MNMS': metrics.dice_coeff_3label,
}


def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _positive_lesion_dice_values(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.ndim == 2:
        pred = pred[None, ...]
        target = target[None, ...]

    dice_values = []
    for idx in range(pred.shape[0]):
        gt = target[idx]
        if not gt.any():
            continue
        pred_slice = pred[idx]
        intersection = int(np.logical_and(pred_slice, gt).sum())
        denominator = int(pred_slice.sum()) + int(gt.sum())
        dice_values.append(_safe_div(2 * intersection, denominator))
    return dice_values


def _lesion_confusion_counts(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.ndim == 2:
        pred = pred[None, ...]
        target = target[None, ...]
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    return tp, fp, fn


def _case_id_from_img_name(img_name):
    base = os.path.splitext(os.path.basename(str(img_name)))[0]
    parts = base.split('_')
    if len(parts) > 1 and parts[0].startswith('Dom'):
        base = '_'.join(parts[1:])
    return base.rsplit('_', 1)[0] if '_' in base else base

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    val_dice = [0.0] * n_part
    domain_metrics = []
    domain_num = len(test_dataloader)
    if args.dataset == 'fundus':
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        softmax, sigmoid, multi = False, True, True
    elif args.dataset in ['prostate', 'prostate_lesion']:
        if args.dataset == 'prostate_lesion' and args.lesion_ce_pos_weight > 0:
            ce_weight = torch.tensor([1.0, args.lesion_ce_pos_weight], dtype=torch.float32).cuda()
            ce_loss = CrossEntropyLoss(weight=ce_weight, reduction='none')
        else:
            ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False
    elif args.dataset == 'MNMS':
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, True
    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        domain_val_dice_count = [0] * n_part
        domain_case_stats = {}
        for batch_num,sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            if args.dataset == 'fundus':
                cup_mask = mask.eq(0).float()
                disc_mask = mask.le(128).float()
                mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)),dim=1)
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            elif args.dataset == 'prostate_lesion':
                mask = mask.gt(0).long()
            elif args.dataset == 'MNMS':
                mask_ = mask[:, ..., 0].eq(255).float()
                mask_[mask[:, ..., 1].eq(255)] = 2
                mask_[mask[:, ..., 2].eq(255)] = 3
                mask = mask_.long()
            output = model(data)
            loss_seg = ce_loss(output, mask).mean() + \
                        dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)

            
            if args.dataset == 'fundus':
                dice = dice_calcu[args.dataset](np.asarray(torch.sigmoid(output.cpu())) >= 0.5, mask.clone().cpu())
                dice_sums = dice
                dice_counts = [1] * len(dice)
            elif args.dataset in ['prostate', 'prostate_lesion', 'MNMS']:
                pred_eval = torch.max(torch.softmax(output.cpu(), dim=1), dim=1)[1]
                if args.dataset == 'prostate_lesion' and (
                    args.post_min_area > 0 or args.post_topk > 0 or args.post_fill_holes
                ):
                    pred_np = util.postprocess_binary_batch(
                        pred_eval.numpy(),
                        min_area=args.post_min_area,
                        topk=args.post_topk,
                        fill_holes=bool(args.post_fill_holes),
                    )
                    pred_eval = torch.from_numpy(pred_np).long()
                if args.dataset == 'prostate_lesion':
                    pred_np = np.asarray(pred_eval)
                    mask_np = mask.clone().cpu().numpy()
                    for sample_idx in range(pred_np.shape[0]):
                        case_id = _case_id_from_img_name(sample['img_name'][sample_idx])
                        if case_id not in domain_case_stats:
                            domain_case_stats[case_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'slices': 0}
                        tp, fp, fn = _lesion_confusion_counts(pred_np[sample_idx], mask_np[sample_idx])
                        domain_case_stats[case_id]['tp'] += tp
                        domain_case_stats[case_id]['fp'] += fp
                        domain_case_stats[case_id]['fn'] += fn
                        domain_case_stats[case_id]['slices'] += 1
                    dice = [0.0]
                    dice_sums = [0.0]
                    dice_counts = [0]
                else:
                    dice = dice_calcu[args.dataset](np.asarray(pred_eval), mask.clone().cpu())
                    dice_sums = dice
                    dice_counts = [1] * len(dice)
            else:
                dice_sums = dice
                dice_counts = [1] * len(dice)
            
            domain_val_loss += loss_seg.item()
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice_sums[i]
                domain_val_dice_count[i] += dice_counts[i]

        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        if args.dataset == 'prostate_lesion':
            case_dice = []
            for case_stat in domain_case_stats.values():
                tp = case_stat['tp']
                fp = case_stat['fp']
                fn = case_stat['fn']
                denominator = 2 * tp + fp + fn
                if denominator > 0:
                    case_dice.append(float(2 * tp) / float(denominator))
            domain_val_dice[0] = _safe_div(sum(case_dice), len(case_dice))
            domain_val_dice_count[0] = len(case_dice)
        for i in range(len(domain_val_dice)):
            if args.dataset != 'prostate_lesion':
                domain_val_dice[i] = _safe_div(domain_val_dice[i], domain_val_dice_count[i])
            val_dice[i] += domain_val_dice[i]
            if args.dataset == 'prostate_lesion':
                writer.add_scalar('{}_val/domain{}/volume_dice_cases'.format(model_name, dc), domain_val_dice_count[i], epoch)
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, dc, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (dc, epoch, domain_val_loss)
        for n, p in enumerate(part):
            if args.dataset == 'prostate_lesion':
                text += ' val_%s_volume_dice: %f' % (p, domain_val_dice[n])
                text += ' volume_cases: %d' % domain_val_dice_count[n]
            else:
                text += ' val_%s_dice: %f' % (p, domain_val_dice[n])
            if n != n_part-1:
                text += ','
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    for n, p in enumerate(part):
        if args.dataset == 'prostate_lesion':
            text += ' val_%s_volume_dice: %f' % (p, val_dice[n])
        else:
            text += ' val_%s_dice: %f' % (p, val_dice[n])
        if n != n_part-1:
            text += ','
    logging.info(text)
    return val_dice

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    if args.dataset == 'fundus':
        num_channels = 3
        patch_size = 256
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        if args.max_iterations is None:
            args.max_iterations = 30000
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        num_channels = 1
        patch_size = 384
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.1, 2
        fillcolor = 255
        if args.max_iterations is None:
            args.max_iterations = 60000
        if args.domain_num >= 6:
            args.domain_num = 6
    elif args.dataset == 'prostate_lesion':
        lesion_modalities = split_csv(args.lesion_modalities) or ['t2w', 'adc']
        num_channels = len(lesion_modalities) + int(args.add_adc_sobel)
        patch_size = 224
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 30000
        if args.domain_num >= 3:
            args.domain_num = 3
        if args.LB == parser.get_default('LB'):
            args.LB = args.lesion_fda_LB
    elif args.dataset == 'MNMS':
        num_channels = 1
        patch_size = 288
        num_classes = 4
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 60000
        if args.domain_num >= 4:
            args.domain_num = 4

    max_iterations = args.max_iterations
    if args.dataset == 'prostate_lesion':
        weak = transforms.Compose([
            tr.LesionAwareScaleCrop(
                patch_size,
                positive_crop_prob=args.lesion_positive_crop_prob,
                center_jitter=args.lesion_crop_jitter,
                mask_fill=0,
                image_fill=0,
                strict=bool(args.lesion_crop_strict),
            ),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
        ])
    else:
        weak = transforms.Compose([
            tr.RandomScaleCrop(patch_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
        ])

    if args.dataset == 'prostate_lesion':
        lesion_modalities = split_csv(args.lesion_modalities) or ['t2w', 'adc']
        strong = transforms.Compose([
            tr.RandomCLAHE(
                p=args.clahe_prob,
                clip_limit=args.clahe_clip,
                tile_grid_size=8,
                blend_alpha=0.7,
            ),
            tr.RandomGamma(p=args.gamma_prob, gamma_min=0.75, gamma_max=1.45),
            tr.RandomBiasField(p=args.bias_prob, strength=0.25),
            tr.RandomGaussianBlurMRI(p=args.blur_prob, radius_min=0.2, radius_max=1.0),
        ])
        normal_toTensor = transforms.Compose([
            tr.LesionNormalize(
                modalities=lesion_modalities,
                mode=args.lesion_norm,
                add_adc_sobel=bool(args.add_adc_sobel),
            ),
            tr.ToTensor()
        ])
    else:
        strong = transforms.Compose([
                tr.Brightness(min_v, max_v),
                tr.Contrast(min_v, max_v),
                tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
        ])
        normal_toTensor = transforms.Compose([
            tr.Normalize_tf(),
            tr.ToTensor()
        ])

    domain_num = args.domain_num
    domain = list(range(1, domain_num + 1))
    primary_dir = None
    primary_modality = None
    if args.dataset == 'prostate_lesion':
        primary_dirs = split_csv(args.lesion_modality_dirs)
        primary_dir = primary_dirs[0] if primary_dirs else None
        primary_modality = (split_csv(args.lesion_modalities) or ['t2w', 'adc'])[0]
    domain_len = get_train_domain_lengths(
        args.dataset, train_data_path, domain_num,
        primary_modality=primary_modality,
        primary_dir=primary_dir,
    )
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain - 1]
    lb_num = args.lb_num
    if lb_num > data_num:
        raise ValueError(f'lb_num ({lb_num}) cannot be greater than number of train images in labeled domain ({data_num}).')
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    dataset_kwargs = {}
    if args.dataset == 'prostate_lesion':
        dataset_kwargs = {
            'modalities': args.lesion_modalities,
            'modality_dirs': args.lesion_modality_dirs,
        }
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor,
                                                **dataset_kwargs)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor,
                                                **dataset_kwargs)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor,
                              **dataset_kwargs)
        test_dataset.append(cur_dataset)
    lb_sampler = make_lesion_sampler(lb_dataset, args.lesion_sampling_prob) if args.dataset == 'prostate_lesion' else None
    ulb_sampler = make_lesion_sampler(ulb_dataset, args.lesion_sampling_prob) if (
        args.dataset == 'prostate_lesion' and args.lesion_sample_unlabeled
    ) else None
    lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=(lb_sampler is None),
                                     sampler=lb_sampler, num_workers=2, pin_memory=True, drop_last=True))
    ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=(ulb_sampler is None),
                                      sampler=ulb_sampler, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        model = build_segmentation_model(
            args.model,
            n_channels=num_channels,
            n_classes=num_classes,
            patch_size=patch_size,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
        )
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train
    if args.dataset == 'fundus':
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        softmax, sigmoid, multi = False, True, True
    elif args.dataset in ['prostate', 'prostate_lesion']:
        if args.dataset == 'prostate_lesion' and args.lesion_ce_pos_weight > 0:
            ce_weight = torch.tensor([1.0, args.lesion_ce_pos_weight], dtype=torch.float32).cuda()
            ce_loss = CrossEntropyLoss(weight=ce_weight, reduction='none')
        else:
            ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False
    elif args.dataset == 'MNMS':
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, True
    dice_loss = losses.DiceLossWithMask(num_classes)
    boundary_loss = losses.BinaryBoundaryLoss(radius=args.boundary_radius)
    focal_loss = losses.FocalLoss(gamma=2, alpha=[0.25, 0.75])

    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)
    threshold = args.threshold

    if args.load:
        path_str = args.load_path
        start_epoch, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter = util.load_osmancheckpoint(
            path_str, ema_model, model, optimizer
        )
        iter_num = start_epoch*args.num_eval_iter
        logging.info('Models restored from epoch {}'.format(start_epoch))

    scaler = GradScaler(device='cuda', enabled=bool(args.amp))
    amp_cm = autocast if args.amp else (lambda **kwargs: contextlib.nullcontext())

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            lesion_modalities = split_csv(args.lesion_modalities) or ['t2w', 'adc']
            fda_indices = select_fda_channel_indices(
                lesion_modalities if args.dataset == 'prostate_lesion' else ['image'] * lb_x_w.shape[1],
                lb_x_w.shape[1],
                args.fda_channels if args.dataset == 'prostate_lesion' else 'all',
            )
            move_transx = []
            for i in range(len(lb_x_w)):
                src_255 = tensor_to_255_np(lb_x_w[i])
                trg_255 = tensor_to_255_np(ulb_x_w[i])
                if len(fda_indices) == 0:
                    img_freq = src_255
                else:
                    amp_trg = extract_amp_spectrum(trg_255)
                    fda_degree = iter_num / max_iterations
                    if args.dataset == 'prostate_lesion':
                        fda_degree = max(float(args.lesion_fda_min_degree), fda_degree)
                    img_freq = source_to_target_freq(src_255, amp_trg, L=args.LB, degree=fda_degree)
                    keep_indices = [idx for idx in range(src_255.shape[0]) if idx not in fda_indices]
                    if keep_indices:
                        img_freq[keep_indices] = src_255[keep_indices]
                img_freq = np.clip(img_freq, 0, 255).astype(np.float32)
                move_transx.append(img_freq)
            move_transx = torch.tensor(np.array(move_transx), dtype=torch.float32)
            move_transx = move_transx/127.5 -1
            move_transx = move_transx.cuda()

            lb_x_w, lb_y, ulb_x_w, ulb_x_s, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_y.cuda()
            if args.dataset == 'fundus':
                lb_cup_label = lb_y.eq(0).float() # == 0
                lb_disc_label = lb_y.le(128).float()  # <= 128
                lb_mask = torch.cat((lb_cup_label.unsqueeze(1), lb_disc_label.unsqueeze(1)),dim=1)
                ulb_cup_label = ulb_y.eq(0).float()
                ulb_disc_label = ulb_y.le(128).float()
                ulb_mask = torch.cat((ulb_cup_label.unsqueeze(1), ulb_disc_label.unsqueeze(1)),dim=1)
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).long()
                ulb_mask = ulb_y.eq(0).long()
            elif args.dataset == 'prostate_lesion':
                lb_mask = lb_y.gt(0).long()
                ulb_mask = ulb_y.gt(0).long()
            elif args.dataset == 'MNMS':
                lb_mask = lb_y[:, ..., 0].eq(255).float()
                lb_mask[lb_y[:, ..., 1].eq(255)] = 2
                lb_mask[lb_y[:, ..., 2].eq(255)] = 3
                lb_mask = lb_mask.long()
                ulb_mask = ulb_y[:, ..., 0].eq(255).float()
                ulb_mask[ulb_y[:, ..., 1].eq(255)] = 2
                ulb_mask[ulb_y[:, ..., 2].eq(255)] = 3
                ulb_mask = ulb_mask.long()

            if args.dataset == 'prostate_lesion' and args.lesion_safe_fda:
                move_transx = protect_frequency_image(move_transx, lb_x_w, lb_mask, args.lesion_cp_dilate)

            with amp_cm(device_type='cuda'):
                with torch.no_grad():
                    if args.dataset == 'prostate_lesion':
                        label_box = obtain_lesion_context_mask(
                            lb_mask,
                            p=args.lesion_cp_prob,
                            dilation=args.lesion_cp_dilate,
                            fallback_random=False,
                        )
                    else:
                        label_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=args.cutmix_prob) for i in range(len(ulb_x_s))], dim=0)
                    img_box = label_box.unsqueeze(1)
                    if args.dataset == 'fundus':
                        label_box = label_box.unsqueeze(1)
                    logits_ulb_x_w = ema_model(ulb_x_w)
                    ulb_x_w_ul = ulb_x_w * (1-img_box) + lb_x_w * img_box
                    logits_w_ul = ema_model(ulb_x_w_ul)
                    ulb_x_w_lu = lb_x_w * (1-img_box) + ulb_x_w * img_box
                    logits_w_lu = ema_model(ulb_x_w_lu)
                    if args.dataset == 'fundus':
                        prob = logits_ulb_x_w.sigmoid()
                        pseudo_label = prob.ge(0.5).float()
                        mask = prob.ge(threshold).float() + prob.le(1-threshold).float()
                        prob_w_ul = logits_w_ul.sigmoid()
                        pseudo_label_w_ul = prob_w_ul.ge(0.5).float()
                        mask_w_ul = prob_w_ul.ge(threshold).float() + prob_w_ul.le(1-threshold).float()
                        prob_w_lu = logits_w_lu.sigmoid()
                        pseudo_label_w_lu = prob_w_lu.ge(0.5).float()
                        mask_w_lu = prob_w_lu.ge(threshold).float() + prob_w_lu.le(1-threshold).float()
                    elif args.dataset in ['prostate', 'prostate_lesion', 'MNMS']:
                        prob_ulb_x_w = torch.softmax(logits_ulb_x_w, dim=1)
                        pseudo_label, mask = pseudo_label_and_mask_from_probs(prob_ulb_x_w, threshold, args.dataset)
                        prob_w_ul = torch.softmax(logits_w_ul, dim=1)
                        pseudo_label_w_ul, mask_w_ul = pseudo_label_and_mask_from_probs(prob_w_ul, threshold, args.dataset)
                        prob_w_lu = torch.softmax(logits_w_lu, dim=1)
                        pseudo_label_w_lu, mask_w_lu = pseudo_label_and_mask_from_probs(prob_w_lu, threshold, args.dataset)

                    mask_w = mask_w_ul * (1-img_box) + mask_w_lu * img_box
                    pseudo_label_w = (pseudo_label_w_ul * (1-label_box) + pseudo_label_w_lu * label_box).long()
                    if args.dataset == 'fundus':
                        pseudo_label_w = pseudo_label_w.float()
                        ensemble = (pseudo_label_w == pseudo_label).float() * mask
                    elif args.dataset in ['prostate', 'prostate_lesion', 'MNMS']:
                        ensemble = (pseudo_label_w == pseudo_label).unsqueeze(1).float() * mask
                    mask_w[ensemble == 0] = 0

                mask_ul, mask_lu = mask.clone(), mask.clone()
                ulb_x_s_ul = ulb_x_s * (1-img_box) + move_transx * img_box
                pseudo_label_ul = (pseudo_label * (1-label_box) + lb_mask * label_box).long()
                mask_ul[img_box.expand(mask_ul.shape) == 1] = 1
                ulb_x_s_lu = move_transx * (1-img_box) + ulb_x_s * img_box
                pseudo_label_lu = (lb_mask * (1-label_box) + pseudo_label * label_box).long()
                if args.dataset == 'fundus':
                    pseudo_label_ul = pseudo_label_ul.float()
                    pseudo_label_lu = pseudo_label_lu.float()
                mask_lu[img_box.expand(mask_lu.shape) == 0] = 1
                # outputs for model
                logits_lb_x_w = model(lb_x_w)
                logits_ulb_x_s_ul = model(ulb_x_s_ul)
                logits_ulb_x_s_lu = model(ulb_x_s_lu)
                logits_ulb_x_s = model(ulb_x_s)

                use_dip_cfet = args.dataset == 'prostate_lesion' and bool(args.dip_cfet_block)
                edge_weight_lb = sobel_edge_weight(lb_x_w, args.dip_edge_weight) if use_dip_cfet else None
                if use_dip_cfet:
                    sup_ce = weighted_ce_loss(ce_loss, logits_lb_x_w, lb_mask, weight_map=edge_weight_lb)
                else:
                    sup_ce = ce_loss(logits_lb_x_w, lb_mask).mean()
                sup_loss = sup_ce + dice_loss(
                    logits_lb_x_w, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi
                )
                if args.dataset == 'prostate_lesion':
                    if args.boundary_loss_weight > 0:
                        sup_loss = sup_loss + args.boundary_loss_weight * boundary_loss(logits_lb_x_w, lb_mask)
                    if args.lesion_focal_weight > 0:
                        sup_loss = sup_loss + args.lesion_focal_weight * focal_loss(logits_lb_x_w, lb_mask)

                dip_loss = torch.zeros((), device=lb_x_w.device)
                if use_dip_cfet and args.dip_sup_weight > 0:
                    lb_x_dip = dip_cfet_augment(
                        lb_x_w,
                        aug_prob=args.dip_aug_prob,
                        gamma_min=args.dip_gamma_min,
                        gamma_max=args.dip_gamma_max,
                        contrast=args.dip_contrast,
                        bias_strength=args.dip_bias_strength,
                        noise_std=args.dip_noise_std,
                    )
                    logits_lb_x_dip = model(lb_x_dip)
                    edge_weight_dip = sobel_edge_weight(lb_x_dip, args.dip_edge_weight)
                    dip_sup_loss = weighted_ce_loss(
                        ce_loss, logits_lb_x_dip, lb_mask, weight_map=edge_weight_dip
                    ) + dice_loss(
                        logits_lb_x_dip, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi
                    )
                    if args.boundary_loss_weight > 0:
                        dip_sup_loss = dip_sup_loss + args.boundary_loss_weight * boundary_loss(logits_lb_x_dip, lb_mask)
                    dip_loss = dip_loss + args.dip_sup_weight * dip_sup_loss

                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))

                edge_weight_ul = sobel_edge_weight(ulb_x_w, args.dip_edge_weight) if use_dip_cfet else None
                unsup_ce_ul = weighted_ce_loss(
                    ce_loss, logits_ulb_x_s_ul, pseudo_label_ul, mask=mask_ul, weight_map=edge_weight_ul
                ) if use_dip_cfet else (ce_loss(logits_ulb_x_s_ul, pseudo_label_ul) * mask_ul.squeeze(1)).mean()
                unsup_loss_ul = unsup_ce_ul + dice_loss(
                    logits_ulb_x_s_ul, pseudo_label_ul.unsqueeze(1), mask=mask_ul, softmax=softmax, sigmoid=sigmoid, multi=multi
                )

                unsup_ce_lu = weighted_ce_loss(
                    ce_loss, logits_ulb_x_s_lu, pseudo_label_lu, mask=mask_lu, weight_map=edge_weight_ul
                ) if use_dip_cfet else (ce_loss(logits_ulb_x_s_lu, pseudo_label_lu) * mask_lu.squeeze(1)).mean()
                unsup_loss_lu = unsup_ce_lu + dice_loss(
                    logits_ulb_x_s_lu, pseudo_label_lu.unsqueeze(1), mask=mask_lu, softmax=softmax, sigmoid=sigmoid, multi=multi
                )

                unsup_ce_s = weighted_ce_loss(
                    ce_loss, logits_ulb_x_s, pseudo_label_w, mask=mask_w, weight_map=edge_weight_ul
                ) if use_dip_cfet else (ce_loss(logits_ulb_x_s, pseudo_label_w) * mask_w.squeeze(1)).mean()
                unsup_loss_s = unsup_ce_s + dice_loss(
                    logits_ulb_x_s, pseudo_label_w.unsqueeze(1), mask=mask_w, softmax=softmax, sigmoid=sigmoid, multi=multi
                )

                if use_dip_cfet and args.dip_cons_weight > 0:
                    ulb_x_dip = dip_cfet_augment(
                        ulb_x_s,
                        aug_prob=args.dip_aug_prob,
                        gamma_min=args.dip_gamma_min,
                        gamma_max=args.dip_gamma_max,
                        contrast=args.dip_contrast,
                        bias_strength=args.dip_bias_strength,
                        noise_std=args.dip_noise_std,
                    )
                    logits_ulb_x_dip = model(ulb_x_dip)
                    prob_ref = torch.softmax(logits_ulb_x_s.detach(), dim=1)
                    prob_dip = torch.softmax(logits_ulb_x_dip, dim=1)
                    cons_map = torch.sum((prob_dip - prob_ref) ** 2, dim=1, keepdim=True)
                    dip_cons_loss = weighted_map_mean(cons_map, mask=mask_w, weight_map=edge_weight_ul)
                    dip_loss = dip_loss + args.dip_cons_weight * dip_cons_loss

                loss = sup_loss + consistency_weight * (unsup_loss_ul + unsup_loss_lu + consistency_weight * unsup_loss_s) + dip_loss

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('train/mask', mask.mean(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/unsup_loss_ul', unsup_loss_ul.item(), iter_num)
            writer.add_scalar('train/unsup_loss_lu', unsup_loss_lu.item(), iter_num)
            writer.add_scalar('train/unsup_loss_s', unsup_loss_s.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/bi_consistency_weight', consistency_weight**2, iter_num)
            writer.add_scalar('train/dip_cfet_loss', dip_loss.item(), iter_num)
            if p_bar is not None:
                p_bar.update()

            if args.dataset == 'fundus':
                p_bar.set_description('iteration %d: loss:%.4f,sup_loss:%.4f, unsup_loss_ul:%f, unsup_loss_lu:%f, cons_w:%.4f,mask_ratio:%.4f' 
                                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss_ul.item(), unsup_loss_lu.item(), consistency_weight, mask.mean()))
            elif args.dataset in ['prostate', 'prostate_lesion', 'MNMS']:
                p_bar.set_description('iteration %d : loss:%.3f, sup_loss:%.3f, unsup_loss_ul:%.3f, unsup_loss_lu:%.3f, unsup_loss_s:%.3f, dip:%.3f, cons_w:%.3f, mask_ratio:%.3f' 
                                    % (iter_num, loss.item(), sup_loss.item(), unsup_loss_ul.item(), unsup_loss_lu.item(), unsup_loss_s.item(), dip_loss.item(), consistency_weight, mask.mean()))

        if p_bar is not None:
            p_bar.close()


        logging.info('test ema model')
        text = ''
        val_dice = test(args, ema_model, test_dataloader, epoch_num+1, writer)
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)
        logging.info('test stu model')
        stu_val_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            torch.save(model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
        text = 'checkpoint.pth'
        checkpoint_path = os.path.join(snapshot_path, text)
        util.save_osmancheckpoint(epoch_num+1, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter, checkpoint_path)
        logging.info('save checkpoint to {}'.format(checkpoint_path))

        
    writer.close()


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    snapshot_path = os.path.join(repo_root, "model", args.dataset, args.save_name) + "/"
    default_data_roots = {
        'fundus': os.path.join(repo_root, 'data', 'Fundus'),
        'prostate': os.path.join(repo_root, 'data', 'Prostate'),
        'prostate_lesion': os.path.abspath(os.path.join(repo_root, '..', '..', 'data', 'Prostate_Lesion3')),
        'MNMS': os.path.join(repo_root, 'data', 'MNMS'),
    }
    train_data_path = getattr(args, 'data_root', None) or default_data_roots[args.dataset]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    source_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(source_dir, snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
