import torch 
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from UNet import UNet
from dataset import PolygonDataset
from data_synthesis import create_synthetic_polygons

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
IMAGE_SIZE = 128
DATA_PATH = Path(__file__).parent / 'dataset'
MODEL_SAVE_PATH = Path(__file__).parent / 'models'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

augment_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

train_dataset = PolygonDataset(
        DATA_PATH + '/training/data.json',
        DATA_PATH + '/training/inputs',
        DATA_PATH + '/training/outputs',
        transform=transform
    )
   
val_dataset = PolygonDataset(
        DATA_PATH + '/validation/data.json',
        DATA_PATH +'/validation/inputs',
        DATA_PATH + '/validation/outputs',
        transform=transform
    )

try:
    synthetic_dataset = PolygonDataset(
            'synthetic/data.json',
            'synthetic/inputs',
            'synthetic/outputs',
            transform=augment_transform,
            augment=True
        )
        
        # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([train_dataset, synthetic_dataset])
    print(f"Combined dataset size: {len(combined_dataset)}")
        
    train_dataset = combined_dataset
except:
      print("Synthetic data not found, using original dataset only")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_colors = len(train_dataset.colors)

model = UNet(num_colors=num_colors).to(device)
optimizer = optim.AdamW(model.parameters(), lr =LEARNING_RATE, weight_decay=0.01)
criterion = nn.MSELoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
train_losses = []
val_losses = []
    
best_val_loss = float('inf')
    
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
    for batch in train_pbar:
        inputs = batch['input'].to(device)
        targets = batch['output'].to(device)
        color_indices = batch['color_idx'].to(device)
            
        optimizer.zero_grad()
        outputs = model(inputs, color_indices)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
        
        # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
        for batch in val_pbar:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            color_indices = batch['color_idx'].to(device)
                
            outputs = model(inputs, color_indices)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
        
    scheduler.step(val_loss)
        
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'color_to_idx': train_dataset.color_to_idx,
            'colors': train_dataset.colors
        }, MODEL_SAVE_PATH + 'best_model.pth')
        
    