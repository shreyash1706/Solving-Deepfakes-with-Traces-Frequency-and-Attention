import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import scipy.fftpack


class BT1(nn.Module):
  def __init__(self, in_channels, out_channels,groups = 1):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,stride = 1 ,kernel_size= 3, padding = 1,groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )

  def forward(self, x):
    return self.conv(x)

class BT2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.block(x) + x

class BT3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)  # No ReLU after addition

class BT4(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
      return self.conv(x)

class SpatialLearner(nn.Module):
  def __init__(self):
    super().__init__()
    self.spatial = nn.Sequential(
        AMTEN(),
        BT1(12,64),
        BT1(64,16),
        BT2(16),
        BT2(16),
        BT2(16),
        BT3(16,32),
        BT3(32,64),
        nn.Sequential(                      # ? wrapping BT3 + CBAM
        BT3(64,128),
        CBAM(128)
        ),
        BT3(128,256)
    )
  def forward(self, x):
    return self.spatial(x)

class FrequencyLearner(nn.Module):
  def __init__(self):
    super().__init__()
    self.freq = nn.Sequential(
        BT1(48,48,groups=4),
        BT1(48,96),
        BT1(96,32),
        BT2(32),
        BT2(32),
        BT2(32),
        BT2(32),
        BT2(32),
        BT3(32,32),
        BT3(32,64),
        nn.Sequential(                      # ? wrapping BT3 + CBAM
        BT3(64,128),
        CBAM(128)
    ),
    )
  def forward(self, x):
    return self.freq(x)
   
class AMTEN(nn.Module):
    def __init__(self):
        super(AMTEN, self).__init__()

        # Conv1: Predict pixel values (3 in -> 3 out)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        # Composite Conv2 + Conv3: F1
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(3)

        # Composite Conv4 + Conv5: F2
        self.conv4 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(6)
        self.conv5 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(6)

       

    def forward(self, x,return_features=False):
        pred = self.conv1(x)
        fmt = pred - x  # Residual (manipulation traces)

#        f1 = F.relu(self.bn3(self.conv3(F.relu(self.bn2(self.conv2(fmt))))))
#        f1_concat = torch.cat([f1, fmt], dim=1)
#
#        f2 = F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(f1_concat))))))
#        freu = torch.cat([f2, f1, fmt], dim=1)  # Final feature set(12 channels)
       
        f1_inter = F.relu(self.bn2(self.conv2(fmt)))
        f1 = F.relu(self.bn3(self.conv3(f1_inter)))
        f1_concat = torch.cat([f1, fmt], dim=1)

        f2_inter = F.relu(self.bn4(self.conv4(f1_concat)))
        f2 = F.relu(self.bn5(self.conv5(f2_inter)))
        freu = torch.cat([f2, f1_concat], dim=1)
        return freu
       
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class AMTENFC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.spatial = SpatialLearner()
        self.freq = FrequencyLearner()
        self.l2norm = nn.LayerNorm([384, 8, 8])
        self.cbam = CBAM(gate_channels=384)              # CBAM added
        self.bt4 = BT4(in_channels=384)
        self.classifier = nn.Linear(384, num_classes)    # Raw logits

    def forward(self, rgb, dct):
        x_spatial = self.spatial(rgb)                    # [B, 256, 8, 8]
        x_freq = self.freq(dct)                          # [B, 128, 8, 8]
        x = torch.cat([x_spatial, x_freq], dim=1)        # [B, 384, 8, 8]
        x = self.l2norm(x)
        x = self.cbam(x)                                 # ?? Apply CBAM
        x = self.bt4(x)                                  # [B, 384, 1, 1]
        x = x.view(x.size(0), -1)                        # [B, 384]
        return self.classifier(x)  





# ----------------------------
# Optimizer and Scheduler Setup
# ----------------------------

def get_optimizer_scheduler(model):
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    return optimizer, scheduler

AMTENFC_transform = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([
    transforms.RandomRotation(0),
    transforms.RandomRotation(90),
    transforms.RandomRotation(180),
    transforms.RandomRotation(270)
]),
    transforms.ToTensor()
])

# ----------------------------
# DCT Feature Extraction Helpers
# ----------------------------

def block_dct(image_np, block_size=4, stride=2):
    h, w, c = image_np.shape
    num_blocks_y = (h - block_size) // stride + 1
    num_blocks_x = (w - block_size) // stride + 1

    zigzag_index = [
        0,  1,  5,  6,
        2,  4,  7, 12,
        3,  8, 11, 13,
        9, 10, 14, 15
    ]

    def compute_dct(channel):
        coeffs = np.zeros((16, num_blocks_y, num_blocks_x))
        for i, y in enumerate(range(0, h - block_size + 1, stride)):
            for j, x in enumerate(range(0, w - block_size + 1, stride)):
                patch = channel[y:y+block_size, x:x+block_size]
                dct_patch =scipy.fftpack.dct(scipy.fftpack.dct(patch.T, norm='ortho').T,norm='ortho')
                coeffs[:, i, j] = dct_patch.flatten()[zigzag_index]
        return coeffs  # [16, H', W']

    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    dct_r, dct_g, dct_b = compute_dct(r), compute_dct(g), compute_dct(b)

    # Reorder channels: R0, G0, B0, R1, G1, B1, ..., R15, G15, B15
    reordered = []
    for i in range(16):
        reordered.extend([dct_r[i], dct_g[i], dct_b[i]])
    output = np.stack(reordered, axis=0)  # [48, H', W']
    return output



# ----------------------------
# Dataset with Real DCT Features
# ----------------------------

class FaceDataset(Dataset):
    def __init__(self, dataframe, root_dir, spatial_transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['image_path'])
        dct_path = self.dataframe.iloc[idx]['dct_path']
        label = self.dataframe.iloc[idx]['label']

        image = Image.open(img_path).convert('RGB').resize((128, 128))

        if self.spatial_transform:
            spatial_image = self.spatial_transform(image)
        else:
            spatial_image = transforms.ToTensor()(image)

        freq_feature = np.load(dct_path)
        freq_feature = torch.from_numpy(freq_feature).float()

        return spatial_image, freq_feature, torch.tensor(label, dtype=torch.float32)


# ----------------------------
# Training Function
# ----------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda:1'):
    best_acc = 0.0
    start_epoch = 0
    checkpoint_path = 'AMTENfc_checkpoint.pth'

    if os.path.exists(checkpoint_path):
        print("Resuming from last checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch} with best acc {best_acc:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for spatial, freq, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            spatial, freq, labels = spatial.to(device), freq.to(device), labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(spatial, freq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spatial.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for spatial, freq, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]'):
                spatial, freq, labels = spatial.to(device), freq.to(device), labels.unsqueeze(1).to(device)

                outputs = model(spatial, freq)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * spatial.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_corrects += torch.sum(preds == labels)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        scheduler.step(val_acc)
       
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)
        print("Checkpoint saved.")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_amtenfc.pth')
            print("Best model saved.")

# ----------------------------
# Main Execution Call
# ----------------------------

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

root_dir = '/home/asds/Deepfake/Datasets/real_vs_fake/real-vs-fake'
def create_dataframe(root_dir, datatype='train'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')

    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    return pd.DataFrame(real_images + fake_images)

train_df = pd.read_csv(os.path.join(root_dir, 'train_dct.csv'))
val_df = pd.read_csv(os.path.join(root_dir, 'valid_dct.csv'))


train_dataset = FaceDataset(train_df, root_dir, spatial_transform=AMTENFC_transform)
val_dataset = FaceDataset(val_df, root_dir, spatial_transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)


model = AMTENFC(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer, scheduler = get_optimizer_scheduler(model)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device=device)
