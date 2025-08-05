import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class PolygonDataset(Dataset):
    def __init__(self, data_json_path, inputs_dir, outputs_dir, transform=None, augment=False):
        with open(data_json_path, 'r') as f:
            self.data = json.load(f)
        
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.transform = transform
        self.augment = augment
        
        # Create color to index mapping
        self.colors = list(set(item['colour'] for item in self.data))
        self.color_to_idx = {color: idx for idx, color in enumerate(self.colors)}
        self.num_colors = len(self.colors)
        
        print(f"Found {len(self.data)} samples with {self.num_colors} unique colors: {self.colors}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image
        input_path = os.path.join(self.inputs_dir, item['input_polygon'])
        input_img = Image.open(input_path).convert('RGB')
        
        # Load output colored polygon image
        output_path = os.path.join(self.outputs_dir, item['output_image'])
        output_img = Image.open(output_path).convert('RGB')
        
        # Get color index
        color_idx = self.color_to_idx[item['colour']]
        
        if self.transform:
            # Apply same transform to both input and output
            seed = np.random.randint(2147483647)
            
            # Transform input
            torch.manual_seed(seed)
            np.random.seed(seed)
            input_tensor = self.transform(input_img)
            
            # Transform output with same seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            output_tensor = self.transform(output_img)
        else:
            to_tensor = transforms.ToTensor()
            input_tensor = to_tensor(input_img)
            output_tensor = to_tensor(output_img)
        
        return {
            'input': input_tensor,
            'output': output_tensor,
            'color_idx': color_idx,
            'color_name': item['colour']
        }