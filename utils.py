import torch
import torch.nn as nn
import numpy as np
from time import time
import albumentations as A
import os
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True)], is_check_shapes=True)


class Metrics():
    def __init__(self, pred, gt, loss_fn, epsilon=1e-7, num_classes=2):
        self.pred = torch.argmax(pred, dim=1)
        self.gt = gt.squeeze(1)
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.pred_ = pred

    def contigious(self, input):
        return input.contiguous().view(-1)
    
    def PA(self):
        with torch.no_grad():
            match = torch.eq(self.pred, self.gt).float()
        return float(match.sum()) / float(match.numel())
    
    def mIOU(self):
        with torch.no_grad():
            pred, gt = self.contigious(self.pred), self.contigious(self.gt)
            iou_per_class = []
            for cls in range(self.num_classes):
                match_pred = pred == cls
                match_gt = gt == cls

                if match_gt.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    inter = torch.logical_and(match_pred, match_gt).sum().float().item()
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()
                    iou = inter / (union + self.epsilon)
                    iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
    
    def loss(self):
        return self.loss_fn(self.pred_, self.gt.long())


def tic_toc(start_time=None):
    return time() - start_time if start_time else time()


class Plot():
    def __init__(self, history, model_name, save_dir=str()):
        self.history = history
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.plot_all()

    def plot_metric(self, metric1, metric2, label1, label2, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history[metric1], label=label1)
        plt.plot(self.history[metric2], label=label2)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{filename}.png"))
        plt.close()


    def plot_all(self):
        self.plot_metric(
            metric1="tr_iou",
            metric2="val_iou",
            label1="Train IOU",
            label2="Validation IOU",
            title=f"Mean Intersection Over Union (mIOU) Score of {self.model_name}",
            ylabel="mIOU Score",
            filename="iou_curve"
        )

        self.plot_metric(
            metric1="tr_pa",
            metric2="val_pa",
            label1="Train Pixel Accuracy",
            label2="Validation Pixel Accuracy",
            title=f"Pixel Accuracy (PA) of {self.model_name}",
            ylabel="Pixel Accuracy",
            filename="pa_curve"
        )

        self.plot_metric(
            metric1="tr_loss",
            metric2="val_loss",
            label1="Train Loss",
            label2="Validation Loss",
            title=f"Loss Curve of{self.model_name}",
            ylabel="Loss",
            filename="loss_curve"
        )


    
