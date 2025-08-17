import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Dataset class
class FaceDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create dataframe from folder
def create_dataframe(root_dir, datatype='train'):
    real_dir = os.path.join(root_dir, datatype, 'real')
    fake_dir = os.path.join(root_dir, datatype, 'fake')
    real_images = [{'image_path': os.path.join(datatype, 'real', f), 'label': 1} for f in os.listdir(real_dir)]
    fake_images = [{'image_path': os.path.join(datatype, 'fake', f), 'label': 0} for f in os.listdir(fake_dir)]
    return pd.DataFrame(real_images + fake_images)
   
class SFFNPlus(nn.Module):
      def __init__(self):
          super(SFFNPlus, self).__init__()
          self.features = nn.Sequential(
              nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 224x224x3 -> 224x224x16
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.MaxPool2d(2),  # -> 112x112
 
              nn.Conv2d(16, 32, kernel_size=3, padding=1),
              nn.BatchNorm2d(32),
              nn.ReLU(),
              nn.MaxPool2d(2),  # -> 56x56
 
              nn.Conv2d(32, 64, kernel_size=3, padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(),
              nn.MaxPool2d(2),  # -> 28x28
 
              nn.Conv2d(64, 128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.MaxPool2d(2),  # -> 14x14
 
              nn.Conv2d(128, 256, kernel_size=3, padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(),
              nn.MaxPool2d(2),  # -> 7x7
 
              nn.Conv2d(256, 512, kernel_size=3, padding=1),
              nn.BatchNorm2d(512),
              nn.ReLU(),
              nn.AdaptiveAvgPool2d((1, 1))  # -> 1x1
          )
 
          self.classifier = nn.Sequential(
              nn.Flatten(),
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Dropout(0.5),
              nn.Linear(256, 1),
              nn.Sigmoid()
          )
 
      def forward(self, x):
          x = self.features(x)
          x = self.classifier(x)
          return x
 
      # Build model
      def build_model(resume=False):
          global checkpoint_path, start_epoch, best_acc
          model = SFFNPlus().to(device)
          optimizer = optim.Adam(model.parameters(), lr=0.001)
          if resume and os.path.exists(checkpoint_path):
              print("Loading checkpoint...")
              checkpoint = torch.load(checkpoint_path)
              model.load_state_dict(checkpoint['model_state_dict'])
              optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
              start_epoch = checkpoint['epoch'] + 1
              best_acc = checkpoint['best_acc']
              print(f"Resuming from epoch {start_epoch} with best acc {best_acc:.4f}")
          return model, optimizer
     
      model, optimizer = build_model(resume=True)
     
      # Training loop
      def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, start_epoch=0, best_acc=0.0):
          for epoch in range(start_epoch, num_epochs):
              print(f'\nEpoch {epoch+1}/{num_epochs}')
              print('-' * 15)
             
              # Training phase
              model.train()
              running_loss, running_corrects = 0.0, 0
              for inputs, labels in tqdm(train_loader, desc='Training'):
                  inputs = inputs.to(device)
                  labels = labels.float().unsqueeze(1).to(device)
     
                  optimizer.zero_grad()
                  outputs = model(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
     
                  running_loss += loss.item() * inputs.size(0)
                  preds = (outputs > 0.5).float()
                  running_corrects += torch.sum(preds == labels.data)
     
              epoch_loss = running_loss / len(train_loader.dataset)
              epoch_acc = running_corrects.double() / len(train_loader.dataset)
     
              # Validation phase
              model.eval()
              val_loss, val_corrects = 0.0, 0
              with torch.no_grad():
                  for inputs, labels in tqdm(val_loader, desc='Validating'):
                      inputs = inputs.to(device)
                      labels = labels.float().unsqueeze(1).to(device)
                      outputs = model(inputs)
                      loss = criterion(outputs, labels)
     
                      val_loss += loss.item() * inputs.size(0)
                      preds = (outputs > 0.5).float()
                      val_corrects += torch.sum(preds == labels.data)
     
              val_loss /= len(val_loader.dataset)
              val_acc = val_corrects.double() / len(val_loader.dataset)
     
              print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
              print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
     
              # Save checkpoint
              torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'best_acc': best_acc
              }, checkpoint_path)
              print("Checkpoint saved.")
     
              # Save best model
              if val_acc > best_acc:
                  best_acc = val_acc
                  torch.save(model.state_dict(), 'best_model_sffn.pth')
                  print("Best model updated.")

if __name__ == "__main__":
   
  # Dataset paths
  root_dir = '/home/user/Deepfake/Datasets/real_vs_fake/real-vs-fake'
  train_df = create_dataframe(root_dir)
  val_df = create_dataframe(root_dir, datatype='valid')
 
  # Dataloaders
  batch_size = 32
  train_dataset = FaceDataset(train_df, root_dir, transform=transform)
  val_dataset = FaceDataset(val_df, root_dir, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
 
  # Loss and checkpoint
  criterion = nn.BCELoss()
  checkpoint_path = 'checkpoint_sffn.pth'
  start_epoch = 0
  best_acc = 0.0
 
  # Enhanced SFFN++
 
 
  train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, start_epoch=start_epoch, best_acc=best_acc)
