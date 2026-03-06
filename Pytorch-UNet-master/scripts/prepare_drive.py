import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare DRIVE for the current U-Net training pipeline')
    parser.add_argument('--source-root', type=Path, required=True, help='Root of the raw DRIVE dataset')
    parser.add_argument('--target-root', type=Path, required=True, help='Output root for prepared DRIVE data')
    return parser.parse_args()


def save_green_channel(image_path: Path, output_path: Path):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    green = image[:, :, 1]
    Image.fromarray(green).save(output_path)


def save_binary_mask(mask_path: Path, output_path: Path):
    mask = np.asarray(Image.open(mask_path).convert('L'))
    binary = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(binary).save(output_path)


def prepare_split(image_paths, manual_paths, target_root: Path):
    images_dir = target_root / 'imgs'
    masks_dir = target_root / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for image_path, manual_path in zip(image_paths, manual_paths):
        sample_name = image_path.stem
        save_green_channel(image_path, images_dir / f'{sample_name}.png')
        save_binary_mask(manual_path, masks_dir / f'{sample_name}.png')


def main():
    args = parse_args()

    images = sorted((args.source_root / 'images').glob('*.tif'))
    manuals = sorted((args.source_root / 'manual').glob('*_manual1.gif'))
    if len(images) != 40 or len(manuals) != 40:
        raise RuntimeError(f'Unexpected DRIVE file count: {len(images)} images, {len(manuals)} masks')

    test_images = images[:20]
    train_images = images[20:]
    test_manuals = manuals[:20]
    train_manuals = manuals[20:]

    prepare_split(train_images, train_manuals, args.target_root / 'train')
    prepare_split(test_images, test_manuals, args.target_root / 'test')

    print(f'Prepared DRIVE at {args.target_root}')


if __name__ == '__main__':
    main()
