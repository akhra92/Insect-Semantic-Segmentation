from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SegmentationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_paths = sorted(glob(f"{root}/images/*.jpg"))
        self.mask_paths = sorted(glob(f"{root}/labels/*.png"))
        self.num_classes = 2
        self.transform = transform        

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img, mask = self.get_img_mask(self.img_paths[idx], self.mask_paths[idx])
        if self.transform:
            img, mask = self.apply_transform(img, mask)

        return img, (mask/255).int()
    
    def get_img_mask(self, img_path, mask_path):
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Convert PIL Images to numpy arrays for albumentations
        img = np.array(img)
        mask = np.array(mask)
        return img, mask
    
    def apply_transform(self, img, mask):
        transformed = self.transform(image=img, mask=mask)
        return transformed["image"], transformed["mask"]
    
def get_dataloaders(root, transform, batch_size, num_workers, split=[0.8, 0.1, 0.1]):
    dataset = SegmentationDataset(root, transform=transform)
    num_classes = dataset.num_classes

    total_size = len(dataset)
    train_size = int(split[0] * total_size)
    val_size = int(split[1] * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, num_classes