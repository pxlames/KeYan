from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loading import BasicDataset
from unet import UNet

ckpt = Path('/root/autodl-tmp/KeYan-lite/Pytorch-UNet-master/checkpoints/checkpoint_epoch50.pth')
images_dir = Path('/root/autodl-tmp/datasets/DRIVE_prepared/test/imgs')
masks_dir = Path('/root/autodl-tmp/datasets/DRIVE_prepared/test/masks')
out_dir = Path('/root/autodl-tmp/KeYan-lite/Pytorch-UNet-master/drive_preds_baseline')
out_dir.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=1, n_classes=1, bilinear=False)
state = torch.load(ckpt, map_location=device)
mask_values = state.pop('mask_values', [0, 1])
model.load_state_dict(state)
model.to(device)
model.eval()

for image_path in sorted(images_dir.glob('*.png'))[:5]:
    mask_path = masks_dir / image_path.name
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    img = BasicDataset.preprocess(mask_values, image, 1.0, is_mask=False)
    x = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)
        pred = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()[0, 0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.array(image), cmap='gray')
    axes[0].set_title('Image')
    axes[1].imshow((np.array(mask) > 0).astype(np.uint8), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / f'{image_path.stem}_triptych.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
print(out_dir)
