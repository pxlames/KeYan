import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('/home/pxl/myProject/血管分割/Pytorch-UNet-master/data/imgs')
dir_mask = Path('/home/pxl/myProject/血管分割/Pytorch-UNet-master/data/masks/')
dir_checkpoint = Path('./checkpoints/')


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _foreground_prob_from_logits(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Return per-pixel foreground probability map with shape [B, H, W]."""
    if n_classes == 1:
        return torch.sigmoid(logits.squeeze(1))
    probs = F.softmax(logits, dim=1)
    fg_idx = 1 if probs.shape[1] > 1 else 0
    return probs[:, fg_idx, ...]


def build_crc_topo_weight(
    logits: torch.Tensor,
    n_classes: int,
    lambda_cal: float,
    temperature: float,
    alpha: float,
    topo_scale: float,
) -> torch.Tensor:
    """
    Build differentiable per-pixel weight map:
      M_f^CRC = relu( sigma((p-lambda_cal)/T) - sigma((p-0.5)/T) )
      W = 1 + alpha * M_f^CRC * M_topo
    where M_topo is a differentiable topological saliency proxy from gradient magnitude.
    """
    p_fg = _foreground_prob_from_logits(logits, n_classes)  # [B, H, W]
    t = max(float(temperature), 1e-6)

    mf_crc = torch.sigmoid((p_fg - lambda_cal) / t) - torch.sigmoid((p_fg - 0.5) / t)
    mf_crc = torch.relu(mf_crc)

    # Differentiable topological proxy: soft edge/centerline saliency via gradients.
    dx = p_fg[..., 1:] - p_fg[..., :-1]
    dy = p_fg[:, 1:, :] - p_fg[:, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-6)

    # Normalize per-sample then squash to (0,1)
    g_mean = grad_mag.mean(dim=(1, 2), keepdim=True)
    g_std = grad_mag.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    m_topo = torch.sigmoid((grad_mag - g_mean) / g_std * topo_scale)

    return 1.0 + alpha * mf_crc * m_topo


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    img = img.unsqueeze(1)
    eroded = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    return eroded.squeeze(1)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    img = img.unsqueeze(1)
    dilated = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    return dilated.squeeze(1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """
    Differentiable soft skeleton approximation adapted from soft-clDice style morphology.
    Input/output shape: [B, H, W], values in [0, 1].
    """
    img = img.clamp(0.0, 1.0)
    skel = torch.relu(img - _soft_open(img))
    for _ in range(max(int(num_iters), 0)):
        img = _soft_erode(img)
        delta = torch.relu(img - _soft_open(img))
        skel = skel + torch.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def degree_distribution_topology_loss(
    logits: torch.Tensor,
    true_masks: torch.Tensor,
    n_classes: int,
    num_iters: int = 20,
    sigma: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compare the skeleton node-degree distributions between prediction and ground truth.
    Returns:
      loss, pred_distribution[B, 5], true_distribution[B, 5]
    Distribution bins correspond to degrees {0, 1, 2, 3, >=4}.
    """
    pred_fg = _foreground_prob_from_logits(logits, n_classes)
    if n_classes == 1:
        true_fg = true_masks.float()
    else:
        true_fg = (true_masks == 1).float()

    pred_skel = soft_skeletonize(pred_fg, num_iters=num_iters)
    true_skel = soft_skeletonize(true_fg, num_iters=num_iters)

    kernel = torch.ones((1, 1, 3, 3), device=logits.device, dtype=pred_skel.dtype)
    kernel[:, :, 1, 1] = 0.0

    pred_deg = F.conv2d(pred_skel.unsqueeze(1), kernel, padding=1).squeeze(1) * pred_skel
    true_deg = F.conv2d(true_skel.unsqueeze(1), kernel, padding=1).squeeze(1) * true_skel

    # Soft histogram bins for degree categories {0,1,2,3,>=4}.
    centers = torch.tensor([0.0, 1.0, 2.0, 3.0], device=logits.device, dtype=pred_skel.dtype).view(1, 4, 1, 1)

    def _soft_degree_hist(degree_map: torch.Tensor, skeleton_map: torch.Tensor) -> torch.Tensor:
        weighted = skeleton_map.unsqueeze(1)
        gaussian_bins = torch.exp(-((degree_map.unsqueeze(1) - centers) ** 2) / max(float(sigma), 1e-6))
        high_degree = torch.sigmoid((degree_map - 3.5) / max(float(sigma), 1e-6)).unsqueeze(1)
        hist = torch.cat([gaussian_bins, high_degree], dim=1) * weighted
        hist = hist.sum(dim=(2, 3))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return hist

    pred_hist = _soft_degree_hist(pred_deg, pred_skel)
    true_hist = _soft_degree_hist(true_deg, true_skel)
    topo_loss = torch.mean(torch.abs(pred_hist - true_hist))
    return topo_loss, pred_hist, true_hist


def train_model(
        model,
        device,
        images_dir,
        masks_dir,
        checkpoint_dir,
        epochs: int = 5,
        start_epoch: int = 0,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_cp_topo_loss: bool = True,
        cp_topo_weight: float = 1.0,
        crc_lambda_cal: float = 0.35,
        cp_temperature: float = 0.05,
        cp_alpha: float = 2.0,
        topo_scale: float = 12.0,
        degree_topo_weight: float = 0.0,
        degree_skeleton_iters: int = 20,
        degree_hist_sigma: float = 0.5,
        num_workers: int = 0,
        crop_size: int = 256,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(images_dir, masks_dir, img_scale, crop_size=crop_size)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(images_dir, masks_dir, img_scale, crop_size=crop_size)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             use_cp_topo_loss=use_cp_topo_loss, cp_topo_weight=cp_topo_weight, crc_lambda_cal=crc_lambda_cal,
             cp_temperature=cp_temperature, cp_alpha=cp_alpha, topo_scale=topo_scale,
             degree_topo_weight=degree_topo_weight, degree_skeleton_iters=degree_skeleton_iters,
             degree_hist_sigma=degree_hist_sigma, checkpoint_dir=str(checkpoint_dir),
             num_workers=num_workers, crop_size=crop_size, start_epoch=start_epoch)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Start epoch:     {start_epoch}
        End epoch:       {start_epoch + epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Crop size:       {crop_size}
        Mixed Precision: {amp}
        CP-Topo loss:    {use_cp_topo_loss}
        CP lambda_cal:   {crc_lambda_cal}
        CP temperature:  {cp_temperature}
        CP alpha:        {cp_alpha}
        Topo scale:      {topo_scale}
        Degree-topo wt:  {degree_topo_weight}
        Degree skel it:  {degree_skeleton_iters}
        Degree sigma:    {degree_hist_sigma}
        Checkpoint dir:  {checkpoint_dir}
        Num workers:     {num_workers}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        current_epoch = start_epoch + epoch
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {current_epoch}/{start_epoch + epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    cp_topo_term = torch.tensor(0.0, device=device)
                    degree_topo_term = torch.tensor(0.0, device=device)
                    if model.n_classes == 1:
                        ce_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        d_loss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        loss = ce_loss + d_loss
                        if use_cp_topo_loss:
                            pixel_ce = F.binary_cross_entropy_with_logits(
                                masks_pred.squeeze(1), true_masks.float(), reduction='none'
                            )
                            w_cp_topo = build_crc_topo_weight(
                                masks_pred, model.n_classes, crc_lambda_cal, cp_temperature, cp_alpha, topo_scale
                            )
                            cp_topo_term = (w_cp_topo * pixel_ce).mean()
                            loss = loss + cp_topo_weight * cp_topo_term
                        if degree_topo_weight > 0:
                            degree_topo_term, _, _ = degree_distribution_topology_loss(
                                masks_pred, true_masks, model.n_classes,
                                num_iters=degree_skeleton_iters, sigma=degree_hist_sigma
                            )
                            loss = loss + degree_topo_weight * degree_topo_term
                    else:
                        ce_loss = criterion(masks_pred, true_masks)
                        d_loss = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        loss = ce_loss + d_loss
                        if use_cp_topo_loss:
                            pixel_ce = F.cross_entropy(masks_pred, true_masks, reduction='none')
                            w_cp_topo = build_crc_topo_weight(
                                masks_pred, model.n_classes, crc_lambda_cal, cp_temperature, cp_alpha, topo_scale
                            )
                            cp_topo_term = (w_cp_topo * pixel_ce).mean()
                            loss = loss + cp_topo_weight * cp_topo_term
                        if degree_topo_weight > 0:
                            degree_topo_term, _, _ = degree_distribution_topology_loss(
                                masks_pred, true_masks, model.n_classes,
                                num_iters=degree_skeleton_iters, sigma=degree_hist_sigma
                            )
                            loss = loss + degree_topo_weight * degree_topo_term

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'cp_topo term': cp_topo_term.item() if use_cp_topo_loss else 0.0,
                    'degree_topo term': degree_topo_term.item() if degree_topo_weight > 0 else 0.0,
                    'step': global_step,
                    'epoch': current_epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': current_epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(Path(checkpoint_dir) / 'checkpoint_epoch{}.pth'.format(current_epoch)))
            logging.info(f'Checkpoint {current_epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Epoch index offset used for resumed training and checkpoint naming')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--checkpoint-dir', type=Path, default=dir_checkpoint,
                        help='Directory used to save checkpoints')
    parser.add_argument('--images-dir', type=Path, default=dir_img, help='Directory containing training images')
    parser.add_argument('--masks-dir', type=Path, default=dir_mask, help='Directory containing training masks')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible runs')
    parser.add_argument('--crop-size', type=int, default=256, help='Random crop size applied to input images')
    parser.add_argument('--disable-cp-topo-loss', action='store_true', default=False,
                        help='Disable CRC-guided differentiable cp-topo weighted CE term')
    parser.add_argument('--cp-topo-weight', type=float, default=1.0,
                        help='Lambda for cp-topo weighted CE term')
    parser.add_argument('--crc-lambda-cal', type=float, default=0.35,
                        help='CRC calibrated threshold lambda_cal (typically < 0.5)')
    parser.add_argument('--cp-temperature', type=float, default=0.05,
                        help='Temperature for smooth CRC uncertainty mask')
    parser.add_argument('--cp-alpha', type=float, default=2.0,
                        help='Scaling factor alpha in cp-topo per-pixel weight')
    parser.add_argument('--topo-scale', type=float, default=12.0,
                        help='Scale used in differentiable topological saliency sigmoid')
    parser.add_argument('--degree-topo-weight', type=float, default=0.0,
                        help='Weight for skeleton node-degree distribution topology loss')
    parser.add_argument('--degree-skeleton-iters', type=int, default=20,
                        help='Number of soft skeletonization iterations for topology statistics')
    parser.add_argument('--degree-hist-sigma', type=float, default=0.5,
                        help='Soft histogram sharpness for skeleton degree distribution loss')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of DataLoader worker processes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    set_random_seed(args.seed)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            start_epoch=args.start_epoch,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_cp_topo_loss=not args.disable_cp_topo_loss,
            cp_topo_weight=args.cp_topo_weight,
            crc_lambda_cal=args.crc_lambda_cal,
            cp_temperature=args.cp_temperature,
            cp_alpha=args.cp_alpha,
            topo_scale=args.topo_scale,
            degree_topo_weight=args.degree_topo_weight,
            degree_skeleton_iters=args.degree_skeleton_iters,
            degree_hist_sigma=args.degree_hist_sigma,
            num_workers=args.num_workers,
            crop_size=args.crop_size,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            start_epoch=args.start_epoch,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_cp_topo_loss=not args.disable_cp_topo_loss,
            cp_topo_weight=args.cp_topo_weight,
            crc_lambda_cal=args.crc_lambda_cal,
            cp_temperature=args.cp_temperature,
            cp_alpha=args.cp_alpha,
            topo_scale=args.topo_scale,
            degree_topo_weight=args.degree_topo_weight,
            degree_skeleton_iters=args.degree_skeleton_iters,
            degree_hist_sigma=args.degree_hist_sigma,
            num_workers=args.num_workers,
            crop_size=args.crop_size,
        )
