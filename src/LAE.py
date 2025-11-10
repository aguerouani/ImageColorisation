import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from skimage.color import rgb2lab, lab2rgb

torch.backends.cudnn.benchmark = True

class RGB2LabDataset(Dataset):
    def __init__(self, image_dir, image_size=256, extensions=('.jpg','.jpeg','.png','.bmp','.webp')):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)
        x_np = x.permute(1,2,0).numpy().astype(np.float32)
        lab = rgb2lab(x_np).astype("float32")

        L  = lab[...,0] / 100.0
        ab = lab[...,1:] / 128.0

        L  = torch.from_numpy(L).unsqueeze(0)
        ab = torch.from_numpy(ab).permute(2,0,1)
        return L, ab

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, down=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 2, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x): 
        return self.block(x)

class LAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1,   32, down=True)
        self.enc2 = ConvBlock(32,  64, down=True)
        self.enc3 = ConvBlock(64, 128, down=True)
        self.enc4 = ConvBlock(128,256, down=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec4 = ConvBlock(256,128, down=False)
        self.dec3 = ConvBlock(128,64,  down=False)
        self.dec2 = ConvBlock(64, 32,  down=False)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x


def lab_to_rgb(L, ab):
    L_ = L[0].detach().cpu().numpy() * 100.0
    ab_ = (ab.detach().cpu().numpy().transpose(1,2,0)) * 128.0
    lab = np.concatenate([L_[...,None], ab_], axis=2).astype(np.float32)
    rgb = lab2rgb(lab)
    return rgb


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0.0
    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()
        ab_pred = model(L)
        loss = criterion(ab_pred, ab)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)
        ab_pred = model(L)
        loss = criterion(ab_pred, ab)
        loss_sum += loss.item()
    return loss_sum / len(loader)


def main():
    data_dir     = "../Data/DIV2K_train_LR_bicubic/X2/"
    image_size   = 256
    batch_size   = 64
    epochs       = 20
    lr           = 2e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = RGB2LabDataset(data_dir, image_size=image_size)
    train_size = int(0.8 * len(dataset))
    eval_size  = len(dataset) - train_size
    train_set, eval_set = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    model = LAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_eval = float('inf')
    os.makedirs("saved_models", exist_ok=True)
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_loss  = evaluate(model, eval_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")

        if eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(model.state_dict(), "saved_models/L_AE_best.pth")

    print("Best Eval Loss:", best_eval)

if __name__ == "__main__":
    main()
