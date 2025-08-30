import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as T


class Inference:
    def __init__(self, model_path, device, save_dir='inference_results'):
        self.device = device
        self.model = smp.Unet(encoder_name='resnet34', encoder_depth=5, classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def tn_2_np(self, t):
        invTrans = T.Compose([
            T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])
        rgb = True if len(t.shape) == 3 else False
        return (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if rgb else (t * 255).detach().cpu().numpy().astype(np.uint8)

    def plot(self, rows, cols, count, im, gt=None, title="Original Image"):
        plt.subplot(rows, cols, count)
        plt.imshow(self.tn_2_np(im.squeeze(0).float()), cmap="gray") if gt else plt.imshow(self.tn_2_np(im.squeeze(0)), cmap="gray")
        plt.axis("off")
        plt.title(title)
        return count + 1

    def run(self, dl, n_ims=15, cols=3, save_name='inference_visualization.png'):
        rows = n_ims // cols
        count = 1
        ims, gts, preds = [], [], []

        for idx, data in enumerate(dl):
            if idx == rows:  # Limit to n_ims/3 samples
                break

            im, gt = data
            with torch.no_grad():
                pred = torch.argmax(self.model(im.to(self.device)), dim=1)

            ims.append(im)
            gts.append(gt)
            preds.append(pred)

        plt.figure(figsize=(25, 20))
        for idx, (im, gt, pred) in enumerate(zip(ims, gts, preds)):
            # Plot original
            count = self.plot(rows, cols, count, im)

            # Plot ground truth
            count = self.plot(rows, cols, count, im=gt.squeeze(0), gt=True, title="Ground Truth")

            # Plot predicted mask
            count = self.plot(rows, cols, count, im=pred, title="Predicted Mask")

        save_path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()