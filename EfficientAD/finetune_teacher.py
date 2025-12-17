#!/usr/bin/python
# -*- coding: utf-8 -*-
import torchvision
import argparse
import os
import random
import copy
import csv
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Wide_ResNet101_2_Weights
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # GUIjW°ƒ(
import matplotlib.pyplot as plt
from common import (get_pdn_small, get_pdn_medium, get_pdn_small_bottleneck, get_pdn_small_dws_small,
                    get_pdn_ghost_simple, get_pdn_small_bottleneckfix, ImageFolderWithoutTarget, InfiniteDataloader)


def get_argparse():
    parser = argparse.ArgumentParser(
        prog='FinetuneTeacher',
        description='Finetune pretrained teacher on MVTec-2 category-specific normal images')
    parser.add_argument('-d', '--mvtec_ad_path', default='./mvtec-2',
                        help='Path to MVTec-2 dataset (default: ./mvtec-2)')
    parser.add_argument('-s', '--subdataset', required=True,
                        help='MVTec category (e.g., bottle, cable, etc.)')
    parser.add_argument('-m', '--model_size',
                        choices=['small', 'medium', 'small_bottleneck', 'small_DWS', 'ghostnet'],
                        default='small',
                        help='Model size (default: small)')
    parser.add_argument('-w', '--pretrained_weights', required=True,
                        help='Path to ImageNet pretrained teacher (e.g., teacher_small_final_state.pth)')
    parser.add_argument('-o', '--output_folder', default='output/finetuning/',
                        help='Output folder (default: output/finetuning/)')
    parser.add_argument('--bottleneck_ratio', type=float, default=2/3,
                        help='Bottleneck compression ratio (default: 2/3)')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Finetuning iterations (default: 5000)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for finetuning (default: 1e-5)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log loss every N iterations (default: 50)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N iterations (default: 1000)')
    parser.add_argument('--val_interval', type=int, default=100,
                        help='Validation loss calculation interval (default: 100)')
    return parser.parse_args()


# variables
seed = 42
on_gpu = torch.cuda.is_available()
device = 'cuda' if on_gpu else 'cpu'

# constants
out_channels = 384
grayscale_transform = transforms.RandomGrayscale(0.1)  # apply same to both
extractor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pdn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_transform(image):
    image = grayscale_transform(image)
    return extractor_transform(image), pdn_transform(image)


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    model_size = config.model_size

    # Output folder (per class)
    output_folder = os.path.join(config.output_folder, config.subdataset)
    os.makedirs(output_folder, exist_ok=True)

    # Backbone (Teacher - Fixed)
    backbone = torchvision.models.wide_resnet101_2(
        weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)

    extractor = FeatureExtractor(backbone=backbone,
                                 layers_to_extract_from=['layer2', 'layer3'],
                                 device=device,
                                 input_shape=(3, 512, 512))

    # PDN (Student - To be finetuned)
    if model_size == 'small':
        pdn = get_pdn_small(out_channels, padding=True)
    elif model_size == 'medium':
        pdn = get_pdn_medium(out_channels, padding=True)
    elif model_size == 'small_DWS':
        pdn = get_pdn_small_dws_small(out_channels, padding=False)
    elif model_size == 'small_bottleneck':
        pdn = get_pdn_small_bottleneck(out_channels, padding=False, bottleneck_ratio=config.bottleneck_ratio)
    elif model_size == 'small_bottleneckfix':
        pdn = get_pdn_small_bottleneckfix(out_channels, padding=False, bottleneck_ratio=config.bottleneck_ratio)
    elif model_size == 'ghostnet':
        pdn = get_pdn_ghost_simple(out_channels, padding=False)
    else:
        raise Exception(f'Unknown model_size: {model_size}')

    # Load pretrained weights
    print(f'Loading pretrained weights from {config.pretrained_weights}')
    ckpt = torch.load(config.pretrained_weights, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        pdn.load_state_dict(ckpt['model_state_dict'])
    else:
        pdn.load_state_dict(ckpt)
    print('Pretrained weights loaded successfully')

    # Data Loading
    mvtec_path = os.path.join(config.mvtec_ad_path, config.subdataset, 'train')
    if not os.path.exists(mvtec_path):
        raise Exception(f'MVTec path not found: {mvtec_path}')

    full_train_set = ImageFolderWithoutTarget(mvtec_path, transform=train_transform)

    # Split train/val
    train_size = int((1 - config.val_split) * len(full_train_set))
    val_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size], generator=rng)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    train_loader_infinite = InfiniteDataloader(train_loader)

    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    print(f'Train set: {train_size} images')
    print(f'Validation set: {val_size} images')

    # Normalization stats calculation on MVTec-2 data
    print('Computing feature normalization statistics on MVTec-2 data...')
    channel_mean, channel_std = feature_normalization(extractor=extractor,
                                                      train_loader=train_loader)

    pdn.train()
    if on_gpu:
        pdn = pdn.cuda()
        channel_mean = channel_mean.cuda()
        channel_std = channel_std.cuda()

    optimizer = torch.optim.Adam(pdn.parameters(), lr=config.lr, weight_decay=1e-5)

    # Scheduler (reduce LR at 95% of training)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(0.95 * config.iterations),
        gamma=0.1
    )

    # Logging
    ratio_suffix = f'_ratio{config.bottleneck_ratio}' if model_size in ['small_bottleneck', 'small_bottleneckfix'] else ''
    log_file = os.path.join(output_folder, f'training_log{ratio_suffix}.csv')
    val_log_file = os.path.join(output_folder, f'validation_log{ratio_suffix}.csv')

    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['iteration', 'train_loss'])
    with open(val_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['iteration', 'val_loss'])

    # Training Loop
    print(f'Starting finetuning for {config.iterations} iterations...')
    tqdm_obj = tqdm(range(config.iterations))
    for iteration, (image_fe, image_pdn) in zip(tqdm_obj, train_loader_infinite):
        if on_gpu:
            image_fe = image_fe.cuda()
            image_pdn = image_pdn.cuda()

        target = extractor.embed(image_fe)
        target = (target - channel_mean) / channel_std
        prediction = pdn(image_pdn)
        prediction = F.interpolate(prediction, size=(64, 64), mode='bilinear', align_corners=False)

        # MSE Loss (same as pretraining)
        loss = torch.mean((target - prediction)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        tqdm_obj.set_description(f'Loss: {loss.item():.6f}, LR: {current_lr:.2e}')

        # Logging
        if iteration % config.log_interval == 0:
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([iteration, loss.item()])

        # Validation
        if iteration % config.val_interval == 0 and iteration > 0:
            val_loss = compute_validation_loss(pdn, val_loader, extractor, channel_mean, channel_std)
            with open(val_log_file, 'a', newline='') as f:
                csv.writer(f).writerow([iteration, val_loss])
            tqdm_obj.write(f'Iteration {iteration}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}')

        # Checkpoint Save
        if iteration % config.save_interval == 0 and iteration > 0:
            checkpoint = {
                'model_state_dict': pdn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'iteration': iteration,
                'channel_mean': channel_mean,  # Essential for inference
                'channel_std': channel_std     # Essential for inference
            }
            torch.save(checkpoint,
                       os.path.join(output_folder,
                                    f'teacher_{model_size}_{config.subdataset}{ratio_suffix}_iter_{iteration}_checkpoint.pth'))

    # Final Save
    checkpoint = {
        'model_state_dict': pdn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iteration': config.iterations,
        'channel_mean': channel_mean,  # Essential for inference
        'channel_std': channel_std     # Essential for inference
    }

    final_ckpt_path = os.path.join(output_folder, f'teacher_{model_size}_{config.subdataset}{ratio_suffix}_final.pth')
    torch.save(checkpoint, final_ckpt_path)

    # Also save state_dict only for compatibility
    torch.save(pdn.state_dict(),
               os.path.join(output_folder,
                            f'teacher_{model_size}_{config.subdataset}{ratio_suffix}_final_state.pth'))

    plot_loss_curve(log_file, val_log_file, output_folder, ratio_suffix)
    print(f'\nFinetuning completed!')
    print(f'Model saved to: {final_ckpt_path}')
    print(f'Log file: {log_file}')
    print(f'Loss curve: {os.path.join(output_folder, f"loss_curve{ratio_suffix}.png")}')


def plot_loss_curve(log_file, val_log_file, output_folder, ratio_suffix):
    """Plot training and validation loss curves"""
    try:
        # Training loss
        train_iterations = []
        train_losses = []
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_iterations.append(int(row['iteration']))
                train_losses.append(float(row['train_loss']))

        # Validation loss
        val_iterations = []
        val_losses = []
        if os.path.exists(val_log_file):
            with open(val_log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val_iterations.append(int(row['iteration']))
                    val_losses.append(float(row['val_loss']))

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train_iterations, train_losses, linewidth=1.5, label='Training Loss', alpha=0.7, color='blue')

        if val_iterations:
            plt.plot(val_iterations, val_losses, linewidth=2, label='Validation Loss',
                    color='red', marker='o', markersize=4, markevery=1)

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Finetuning Loss Curve', fontsize=14)
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_folder, f'loss_curve{ratio_suffix}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Loss curve saved: {plot_path}')
    except Exception as e:
        print(f'Failed to plot loss curve: {e}')


@torch.no_grad()
def compute_validation_loss(pdn, val_loader, extractor, channel_mean, channel_std):
    """Compute validation loss"""
    pdn.eval()
    val_losses = []

    for (image_fe, image_pdn) in val_loader:
        if on_gpu:
            image_fe = image_fe.cuda()
            image_pdn = image_pdn.cuda()

        target = extractor.embed(image_fe)
        target = (target - channel_mean) / channel_std
        prediction = pdn(image_pdn)
        prediction = F.interpolate(prediction, size=(64, 64), mode='bilinear', align_corners=False)
        loss = torch.mean((target - prediction)**2)
        val_losses.append(loss.item())

    pdn.train()
    return np.mean(val_losses)


@torch.no_grad()
def feature_normalization(extractor, train_loader, steps=1000):
    """Compute feature normalization statistics on MVTec-2 data"""
    mean_outputs = []
    normalization_count = 0
    with tqdm(desc='Computing mean of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            if on_gpu:
                image_fe = image_fe.cuda()
            output = extractor.embed(image_fe)
            mean_output = torch.mean(output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    normalization_count = 0
    with tqdm(desc='Computing variance of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            if on_gpu:
                image_fe = image_fe.cuda()
            output = extractor.embed(image_fe)
            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


# FeatureExtractor and related classes (from pretraining.py)
class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, input_shape):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.input_shape = input_shape
        self.patch_maker = PatchMaker(3, stride=1)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(feature_dimensions, 1024)
        self.forward_modules["preprocessing"] = preprocessing

        preadapt_aggregator = Aggregator(target_dim=out_channels)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.forward_modules.eval()

    @torch.no_grad()
    def embed(self, images):
        """Returns feature embeddings for images."""
        _ = self.forward_modules["feature_aggregator"].eval()
        features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1],
                *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        features = torch.reshape(features, (-1, 64, 64, out_channels))
        features = torch.permute(features, (0, 3, 1, 2))

        return features


class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches."""
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


if __name__ == '__main__':
    main()
