import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from prob_unet import ProbabilisticUnet
from prob_unet.drive_dataset import DrivePreparedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export DRIVE triptych predictions for Probabilistic U-Net")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    net = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim=ckpt_args.get("latent_dim", 6),
        no_convs_fcomb=4,
        beta=ckpt_args.get("beta", 10.0),
    )
    net.to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    dataset = DrivePreparedDataset(
        args.data_root,
        split="test",
        scale=args.scale,
        crop_size=0,
        random_crop=False,
        return_id=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for patch, mask, sample_ids in loader:
        patch = patch.to(device)
        mask = mask.to(device)
        net.forward(patch, None, training=False)
        logits = torch.stack([net.sample(testing=True) for _ in range(args.num_samples)], dim=0).mean(dim=0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        for idx, sample_id in enumerate(sample_ids):
            image_np = patch[idx, 0].detach().cpu().numpy()
            mask_np = mask[idx].detach().cpu().numpy()
            pred_np = preds[idx, 0].detach().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(image_np, cmap="gray")
            axes[0].set_title("Image")
            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("GT")
            axes[2].imshow(pred_np, cmap="gray")
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            fig.savefig(output_dir / f"{sample_id}_triptych.png", dpi=160, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
