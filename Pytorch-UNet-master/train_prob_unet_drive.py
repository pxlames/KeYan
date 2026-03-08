import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from prob_unet import ProbabilisticUnet
from prob_unet.drive_dataset import DrivePreparedDataset
from prob_unet.utils import l2_regularisation


def parse_args():
    parser = argparse.ArgumentParser(description="Train Probabilistic U-Net on DRIVE_prepared")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def split_indices(dataset_size, val_ratio, seed):
    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(dataset_size * val_ratio)))
    return indices[val_size:].tolist(), indices[:val_size].tolist()


@torch.no_grad()
def predict_probabilities(net, patch, num_samples):
    net.forward(patch, None, training=False)
    logits = [net.sample(testing=True) for _ in range(num_samples)]
    return torch.sigmoid(torch.stack(logits, dim=0).mean(dim=0))


@torch.no_grad()
def evaluate(net, loader, device, num_samples):
    net.eval()
    stats = {"loss": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "count": 0}
    eps = 1e-8

    for patch, mask in loader:
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask, 1)

        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = (
            l2_regularisation(net.posterior)
            + l2_regularisation(net.prior)
            + l2_regularisation(net.fcomb.layers)
        )
        loss = -elbo + 1e-5 * reg_loss

        probs = predict_probabilities(net, patch, num_samples)
        preds = (probs >= 0.5).float()

        tp = (preds * mask).sum(dim=(1, 2, 3))
        fp = (preds * (1.0 - mask)).sum(dim=(1, 2, 3))
        fn = ((1.0 - preds) * mask).sum(dim=(1, 2, 3))

        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)

        batch_size = patch.size(0)
        stats["loss"] += loss.item()
        stats["dice"] += dice.mean().item() * batch_size
        stats["precision"] += precision.mean().item() * batch_size
        stats["recall"] += recall.mean().item() * batch_size
        stats["count"] += batch_size

    count = max(1, stats["count"])
    return {
        "loss": stats["loss"] / max(1, len(loader)),
        "dice": stats["dice"] / count,
        "precision": stats["precision"] / count,
        "recall": stats["recall"] / count,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_train = DrivePreparedDataset(args.data_root, split="train", image_size=args.image_size)
    train_indices, val_indices = split_indices(len(full_train), args.val_ratio, args.seed)
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim=args.latent_dim,
        no_convs_fcomb=4,
        beta=args.beta,
    )
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        running_loss = 0.0

        for step, (patch, mask) in enumerate(train_loader):
            patch = patch.to(device)
            mask = mask.to(device)
            mask = torch.unsqueeze(mask, 1)

            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = (
                l2_regularisation(net.posterior)
                + l2_regularisation(net.prior)
                + l2_regularisation(net.fcomb.layers)
            )
            loss = -elbo + 1e-5 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(net, val_loader, device, args.num_samples)
        record = {"epoch": epoch, "train_loss": running_loss / max(1, len(train_loader)), **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=True))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "metrics": record,
        }
        torch.save(checkpoint, save_dir / "last.pt")

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(checkpoint, save_dir / "best.pt")

        if epoch % args.checkpoint_every == 0:
            torch.save(checkpoint, save_dir / f"epoch_{epoch:03d}.pt")

    with (save_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()

