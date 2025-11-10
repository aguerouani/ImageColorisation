import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2lab, lab2rgb
from torch import nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.utils.data import random_split
import torch.optim as optim
from torchsummary import summary


class RGB2LabDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Resize + Convert to tensor
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # output range: [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)  # (3, H, W)

        # Convert to (H, W, 3) for rgb2lab
        img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)

        # Convert to Lab
        lab = rgb2lab(img_np).astype("float32")  # L in [0,100], a,b ~ [-128,127]

        # Extract channels
        L = lab[:, :, 0]   # (H, W)
        ab = lab[:, :, 1:] # (H, W, 2)

        # Normalize:
        L = L / 100.0
        ab = ab / 128.0

        # Convert to tensors
        L = torch.from_numpy(L).unsqueeze(0)   # (1, H, W)
        ab = torch.from_numpy(ab).permute(2, 0, 1)  # (2, H, W)

        return L, ab
    
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 3),    # (254, 254, 3)
            nn.MaxPool2d(2),       # (127, 127, 3)
            nn.Conv2d(3, 9, 3),    # (125, 125, 9)
            nn.MaxPool2d(2),       # (62, 62, 9)
            nn.Conv2d(9, 27, 3),   # (60, 60, 27)
            nn.MaxPool2d(2),       # (30, 30, 27)
            nn.Conv2d(27, 81, 3),  # (28, 28, 81)
            nn.MaxPool2d(2),       # (14, 14, 81)
            nn.Conv2d(81, 243, 3), # (12, 12, 243)
            nn.MaxPool2d(2),       # (6, 6, 243) # Corrected output size after pooling
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(243, 243, kernel_size=2, stride=2, padding=0, output_padding=0),  # 6 -> 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(243, 81,  kernel_size=3, stride=1, padding=0),                    # 12 -> 14
            nn.ReLU(inplace=True),

            # 14 -> 28 -> 30
            nn.ConvTranspose2d(81,  81,  kernel_size=2, stride=2, padding=0, output_padding=0),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(81,  27,  kernel_size=3, stride=1, padding=0),                    # 28 -> 30
            nn.ReLU(inplace=True),

            # 30 -> 60 -> 62
            nn.ConvTranspose2d(27,  27,  kernel_size=2, stride=2, padding=0, output_padding=0),  # 30 -> 60
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(27,  9,   kernel_size=3, stride=1, padding=0),                    # 60 -> 62
            nn.ReLU(inplace=True),

            # 62 -> 125 -> 127   (need output_padding=1 to recover odd 125)
            nn.ConvTranspose2d(9,   9,   kernel_size=2, stride=2, padding=0, output_padding=1),  # 62 -> 125
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(9,   3,   kernel_size=3, stride=1, padding=0),                    # 125 -> 127
            nn.ReLU(inplace=True),

            # 127 -> 254 -> 256
            nn.ConvTranspose2d(3,   3,   kernel_size=2, stride=2, padding=0, output_padding=0),  # 127 -> 254
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3,   1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() 
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for L, ab in loader:
        L = L.to(device)
        ab = ab.to(device)
        optimizer.zero_grad()
        outputs = model(L)
        loss = criterion(outputs, L)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for L, ab in loader:
            L = L.to(device)
            ab = ab.to(device)
            outputs = model(L)
            loss = criterion(outputs, L)
            running_loss += loss.item()
    return running_loss / len(loader)

def lab_to_rgb(L, ab):
    """
    L is (1,H,W), ab is (2,H,W), both torch tensors.
    L is in [0,1], ab is in [-1,1].
    We convert them back to LAB then RGB.
    """
    L = L[0].cpu().numpy()          # (H,W)
    ab = ab.cpu().numpy().transpose(1,2,0)  # (H,W,2)

    # Undo normalization
    L = L * 100
    ab = ab * 128

    lab = np.concatenate([L[..., np.newaxis], ab], axis=2)  # (H,W,3)
    rgb = lab2rgb(lab)  # returns floats in [0,1]
    return rgb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = RGB2LabDataset("../Data/DIV2K_train_LR_bicubic/X2/", image_size=256)

    # Split dataset into training and evaluation sets
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    
    model = AE().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model = AE().to(device)
    summary(model, input_size=(1, 256, 256), device=str(device))
    
    num_epochs = 10 # You can adjust the number of epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        eval_loss = evaluate(model, eval_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        
    model.eval()
    L_batch, ab_batch = next(iter(eval_loader))
    L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
    with torch.no_grad():
        pred_L_batch = model(L_batch)
    
    num_show = 5
    plt.figure(figsize=(12, num_show * 3))

    for i in range(num_show):
        L = L_batch[i]
        ab_gt = ab_batch[i]
        L_pred = pred_L_batch[i]

        # Convert to RGB
        rgb_gt = lab_to_rgb(L, ab_gt)
        rgb_pred = lab_to_rgb(L_pred, ab_gt)

        # Input grayscale for visualization
        grayscale = L[0].cpu().numpy()

        # Plot
        plt.subplot(num_show, 3, i*3 + 1)
        plt.imshow(grayscale, cmap='gray')
        plt.title("Input L")
        plt.axis('off')

        plt.subplot(num_show, 3, i*3 + 2)
        plt.imshow(rgb_gt)
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_show, 3, i*3 + 3)
        plt.imshow(rgb_pred)
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()