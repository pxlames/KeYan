import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from prob_unet import ProbabilisticUnet
from prob_unet.drive_dataset import DrivePreparedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Probabilistic U-Net on DRIVE_prepared/test")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        args.data_root, split="test", image_size=args.image_size, return_id=True
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    eps = 1e-8
    dice_scores = []
    precision_scores = []
    recall_scores = []

    for patch, mask, sample_ids in loader:
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask, 1)

        net.forward(patch, None, training=False)
        logits = torch.stack([net.sample(testing=True) for _ in range(args.num_samples)], dim=0).mean(dim=0)
        preds = (torch.sigmoid(logits) >= 0.5).float()

        tp = (preds * mask).sum(dim=(1, 2, 3))
        fp = (preds * (1.0 - mask)).sum(dim=(1, 2, 3))
        fn = ((1.0 - preds) * mask).sum(dim=(1, 2, 3))

        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        for idx, sample_id in enumerate(sample_ids):
            print(
                json.dumps(
                    {
                        "id": sample_id,
                        "dice": float(dice[idx]),
                        "precision": float(precision[idx]),
                        "recall": float(recall[idx]),
                    },
                    ensure_ascii=True,
                )
            )

        dice_scores.extend(dice.cpu().numpy().tolist())
        precision_scores.extend(precision.cpu().numpy().tolist())
        recall_scores.extend(recall.cpu().numpy().tolist())

    print(
        json.dumps(
            {
                "summary": {
                    "dice": float(np.mean(dice_scores)),
                    "precision": float(np.mean(precision_scores)),
                    "recall": float(np.mean(recall_scores)),
                }
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
