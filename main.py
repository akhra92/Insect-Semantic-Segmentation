import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import argparse
from utils import get_transforms
from dataset import get_dataloaders
from torch.optim import Adam
from train import Trainer
from utils import Plot
from test import Inference


parser = argparse.ArgumentParser(description="Segmentation Project")
parser.add_argument("-bs", type=int, default=8, help="Batch Size")
parser.add_argument("-lr", type=float, default=1e-3, help="Learning Rate")
parser.add_argument("-ep", type=int, default=10, help="Number of Epochs")
parser.add_argument("-d", type=str, default="cpu", help="Device ['cuda' or 'cpu']")
parser.add_argument("-rt", type=str, default="./datasets/insect_semantic_segmentation/arthropodia", help="Root of the dataset")
parser.add_argument("-nw", type=int, default=2, help="Number of workers")
parser.add_argument("-mp", type=str, default="saved_models/insect_best_model.pt", help="Path of the saved model")
parser.add_argument("-nc", type=int, default=2, help="Number of classes")

args = parser.parse_args()

transform = get_transforms(img_size=256)
trn_loader, val_loader, ts_loader, num_classes = get_dataloaders(root=args.rt, transform=transform, batch_size=args.bs, num_workers=args.nw)
model = smp.Unet(encoder_name="resnet34", encoder_depth=5, classes=args.nc)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=args.lr)

trainer = Trainer(model=model,
                  device=args.d,
                  train_loader=trn_loader,
                  val_loader=val_loader,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  num_classes=num_classes)

if __name__ == "__main__":
    history = trainer.run(epochs=args.ep, save_prefix="insect")
    Plot(history)
    inference_runner = Inference(model_path=args.mp, device=args.d)
    inference_runner.run(dl=ts_loader)