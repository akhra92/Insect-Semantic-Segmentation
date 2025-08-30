import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from utils import Metrics


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, loss_fn, num_classes, save_dir="saved_models", early_stop_thresh=5, thresh=0.005):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.early_stop_thresh = early_stop_thresh
        self.thresh = thresh
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def run(self, epochs, save_prefix):
        self.model.to(self.device)
        tr_loss, tr_pa, tr_miou = [], [], []
        val_loss, val_pa, val_miou = [], [], []
        best_loss, not_improve = np.inf, 0

        print("Starting Training Process...")
        for epoch in range(1, epochs+1):
            print(f"\nEpoch {epoch}/{epochs}")
            self.model.train()
            train_metrics = self._process_epoch(self.train_loader, is_training=True)

            self.model.eval()
            with torch.no_grad():
                val_metrics = self._process_epoch(self.val_loader, is_training=False)

            tr_loss.append(train_metrics["loss"])
            tr_pa.append(train_metrics["pa"])
            tr_miou.append(train_metrics["iou"])
            val_loss.append(val_metrics["loss"])
            val_pa.append(val_metrics["pa"])
            val_miou.append(val_metrics["iou"])

            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f'Train Loss: {train_metrics["loss"]} | Train PA: {train_metrics["pa"]} | Train mIOU: {train_metrics["iou"]}')
            print(f'Val Loss: {val_metrics["loss"]} | Val PA: {val_metrics["pa"]} | Val mIOU: {val_metrics["iou"]}')
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            if best_loss > (val_metrics["loss"] + self.thresh):
                print(f'Loss has decreased from {best_loss} to {val_metrics["loss"]}')
                best_loss = val_metrics["loss"]
                not_improve = 0
                torch.save(self.model.state_dict(), f"{self.save_dir}/{save_prefix}_best_model.pt")
            else:
                not_improve += 1
                if not_improve >= self.early_stop_thresh:
                    print(f"Early Stopping! Loss has not changed for {not_improve} times")
                    break
        
        return {"tr_loss": tr_loss, "tr_pa": tr_pa, "tr_miou": tr_miou,
                "val_loss": val_loss, "val_pa": val_pa, "val_miou": val_miou}
    
    def _process_epoch(self, dataloader, is_training):
        phase = "Train" if is_training else "Validation"
        print(f"{phase} have started!")
        total_loss, total_pa, total_miou = 0, 0, 0

        for img, gt in tqdm(dataloader, desc=f"{phase} Progress!"):
            img, gt = img.to(self.device), gt.to(self.device)
            if is_training:
                preds = self.model(img)
                metrics = Metrics(preds, gt, self.loss_fn)
                loss = metrics.loss()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                preds = self.model(img)
                metrics = Metrics(preds, gt, self.loss_fn)
                loss = metrics.loss()
            
            total_loss += loss.item()
            total_miou += metrics.mIOU()
            total_pa += metrics.PA()

        num_batches = len(dataloader)
        return {"loss": total_loss / num_batches,
                "pa": total_pa / num_batches,
                "iou": total_miou / num_batches}

            