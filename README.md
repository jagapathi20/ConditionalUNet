# ConditionalUNet

A PyTorch implementation of Conditional UNet for **polygon coloring**. This model takes polygon outlines as input and generates colored polygons based on specified color conditions. The project demonstrates conditional image generation using a UNet architecture with color embeddings.

## 🎯 Project Overview

This ConditionalUNet learns to:
- Take black and white polygon outlines as input
- Generate colored polygons based on color conditions (red, blue, green, yellow, purple, orange, cyan, magenta)
- Handle various polygon shapes (triangles to octagons)
- Support both real and synthetic training data

## 🚀 Features

- **Conditional Color Generation**: Generate polygons in 8 different colors
- **UNet Architecture**: Classic encoder-decoder with skip connections
- **Color Embeddings**: Learned embeddings for color conditions
- **Synthetic Data Generation**: Automatic creation of training data
- **Data Augmentation**: Rotation and flipping for better generalization
- **Model Checkpointing**: Automatic saving of best models

## 📋 Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy
opencv-python
pillow
tqdm
pathlib
json
```

## 🔧 Installation

### Option 1: Local Setup
```bash
# Clone the repository
git clone https://github.com/jagapathi20/ConditionalUNet.git
cd ConditionalUNet

# Install dependencies
pip install torch torchvision numpy opencv-python pillow tqdm
```

### Option 2: Google Colab (Recommended for Beginners)
Click the Colab link in the Usage section - no installation required!

## 📊 Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── training/
│   ├── data.json
│   ├── inputs/          # Polygon outline images
│   └── outputs/         # Colored polygon images
└── validation/
    ├── data.json
    ├── inputs/
    └── outputs/
```

### Data JSON Format
```json
[
  {
    "input_polygon": "input_001.png",
    "output_image": "output_001.png", 
    "colour": "red"
  }
]
```

## 📦 Pre-trained Model

Download the pre-trained model weights:
**[🔗 Download Trained Model](https://drive.google.com/file/d/11t5PHTNSSkouYGKjBtNVMz3dCqp7AOpK/view?usp=sharing)**

## 🎮 Usage

### 🚀 Quick Start with Google Colab

For easy experimentation without local setup:

**[📓 Open in Google Colab](https://colab.research.google.com/drive/1RJK4hH7fe0Hp-ajr0wX8-goIff3Bhei8?usp=sharing)**

The Colab notebook includes:
- Complete setup and installation
- Dataset loading and preprocessing
- Model training with visualization
- Inference examples and results
- Interactive experimentation

### Training the Model Locally

```bash
# Train with your dataset
python main.py
```

The training script will:
- Load your polygon dataset
- Generate synthetic data if not available
- Train the ConditionalUNet with color embeddings
- Save the best model based on validation loss

### Inference

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from UNet import UNet

# Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('models/best_model.pth', map_location=device)

model = UNet(num_colors=8).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load polygon outline
input_image = Image.open('path_to_polygon_outline.png').convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Choose color (0-7: red, blue, green, yellow, purple, orange, cyan, magenta)
color_idx = torch.tensor([0]).to(device)  # red

# Generate colored polygon
with torch.no_grad():
    colored_polygon = model(input_tensor, color_idx)
    
# Convert back to image
output_image = transforms.ToPILImage()(colored_polygon.cpu().squeeze(0))
output_image.save('colored_polygon.png')
```

### Generate Synthetic Data

```python
from data_synthesis import create_synthetic_polygons

# Generate 1000 synthetic polygon pairs
create_synthetic_polygons(num_samples=1000, image_size=128)
```

## 🏗️ Model Architecture

The ConditionalUNet consists of:

### Core Components:
- **Color Embedding Layer**: Maps color indices to 64-dimensional embeddings
- **Encoder Path**: 4 downsampling blocks with DoubleConv layers
- **Bottleneck**: Feature processing at the lowest resolution
- **Decoder Path**: 4 upsampling blocks with skip connections
- **Output Layer**: Final 1x1 convolution to RGB output

### Key Features:
- **Conditional Input**: Color embeddings are concatenated as additional input channel
- **Skip Connections**: Preserve fine-grained details from encoder
- **Batch Normalization**: Stable training with BatchNorm layers
- **Flexible Architecture**: Supports different input/output channels

## 📈 Training Details

- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Loss Function**: MSE Loss
- **Batch Size**: 32
- **Image Size**: 128x128
- **Epochs**: 100
- **Colors Supported**: 8 (red, blue, green, yellow, purple, orange, cyan, magenta)

## 📁 File Structure

```
ConditionalUNet/
├── UNet.py              # Main UNet model definition
├── UNet_parts.py        # UNet building blocks (DoubleConv, DownSample, UpSample)
├── dataset.py           # PolygonDataset class for data loading
├── main.py              # Training script
├── data_synthesis.py    # Synthetic data generation
└── models/              # Saved model checkpoints
```

## 🎨 Supported Colors

The model supports 8 different colors:
1. **Red** (255, 0, 0)
2. **Blue** (0, 0, 255) 
3. **Green** (0, 255, 0)
4. **Yellow** (255, 255, 0)
5. **Purple** (128, 0, 128)
6. **Orange** (255, 165, 0)
7. **Cyan** (0, 255, 255)
8. **Magenta** (255, 0, 255)

## 🔬 Model Performance

The model uses:
- **Training Loss**: MSE between generated and target colored polygons
- **Validation Loss**: Monitors generalization performance
- **Learning Rate Scheduling**: Reduces LR on validation plateau
- **Best Model Saving**: Automatically saves best performing checkpoint

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:
- Support for more colors
- Different polygon shapes
- Advanced conditioning mechanisms
- Better loss functions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UNet architecture based on the original paper by Ronneberger et al.
- Conditional generation techniques for controlled image synthesis
- PyTorch community for excellent deep learning framework

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{conditionalunet2024,
  title={ConditionalUNet: Polygon Coloring with Conditional Image Generation},
  author={Jagapathi},
  year={2024},
  url={https://github.com/jagapathi20/ConditionalUNet}
}
```