import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a segmentation checkpoint on a paired image/mask set')
    parser.add_argument('--checkpoint', required=True, help='Path to the checkpoint .pth file')
    parser.add_argument('--images-dir', required=True, help='Directory containing input images')
    parser.add_argument('--masks-dir', required=True, help='Directory containing ground-truth masks')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scaling factor used during training')
    parser.add_argument('--classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--threshold', type=float, default=0.5, help='Foreground threshold for binary segmentation')
    return parser.parse_args()


def load_gray_image(path: Path, scale: float, is_mask: bool) -> torch.Tensor:
    image = Image.open(path).convert('L')
    width, height = image.size
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    image = image.resize(
        (new_width, new_height),
        resample=Image.NEAREST if is_mask else Image.BICUBIC,
    )
    array = np.asarray(image, dtype=np.float32)
    if is_mask:
        return torch.from_numpy((array > 127).astype(np.float32))
    return torch.from_numpy((array / 255.0)[None, ...])


def collect_pairs(images_dir: Path, masks_dir: Path):
    pairs = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.name.startswith('.'):
            continue
        candidates = list(masks_dir.glob(image_path.stem + '.*'))
        if candidates:
            pairs.append((image_path, candidates[0]))
    if not pairs:
        raise RuntimeError(f'No matched image/mask pairs found in {images_dir} and {masks_dir}')
    return pairs


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'mask_values' in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != 'mask_values'}
    model.load_state_dict(state_dict)
    model.eval()

    pairs = collect_pairs(Path(args.images_dir), Path(args.masks_dir))

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for image_path, mask_path in pairs:
            image = load_gray_image(image_path, args.scale, is_mask=False)
            mask_true = load_gray_image(mask_path, args.scale, is_mask=True)

            image = image.unsqueeze(0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device)

            logits = model(image)
            if args.classes == 1:
                mask_pred = (torch.sigmoid(logits.squeeze(0).squeeze(0)) > args.threshold).float()
            else:
                mask_pred = logits.argmax(dim=1).squeeze(0).float()

            tp = float((mask_pred * mask_true).sum().item())
            fp = float((mask_pred * (1.0 - mask_true)).sum().item())
            fn = float(((1.0 - mask_pred) * mask_true).sum().item())

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_dice += (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)

    precision = (total_tp + 1e-6) / (total_tp + total_fp + 1e-6)
    recall = (total_tp + 1e-6) / (total_tp + total_fn + 1e-6)
    dice = total_dice / len(pairs)

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'n_test': len(pairs),
        'dice': dice,
        'precision': precision,
        'recall': recall,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
