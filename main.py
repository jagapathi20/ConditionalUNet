import torch 
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from UNet import UNet
from dataset import PolygonDataset

LEARNING_RATE = 3E-4
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_colors = len(val_dataset.colors)

model = UNet(num_colors=num_colors).to(device)
optimizer = optim.AdamW(model.prameters(), lr =LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
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
        
    scheduler.step()
        
    
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
        
    