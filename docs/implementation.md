# Implementation Guide

## Research Project Overview

This implementation guide covers the research project "Enhancing CNN Robustness By Using Data Augmentation and Oscillation Block" (Luo et al., 2024), focusing on how structural modifications can improve CNNs for image classification tasks.

## Research Objectives

- Incorporate oscillation blocks to capture tilted features in rotated images
- Apply preprocessing techniques (rotation, zoom, positional adjustments) to simulate real-world variations
- Evaluate performance trade-offs between standard and enhanced CNN architectures
- Demonstrate robustness improvements particularly for rotated data

## Setup and Installation

### Requirements

```bash
pip install torch torchvision numpy scikit-learn tqdm Pillow matplotlib
```

### Project Structure

```text
RDENET-Oscillation-Block/
├── src/                    # Source code
├── data/                   # Dataset storage
├── outputs/               # Training results
├── docs/                  # Documentation
└── README.md             # Main documentation
```

## Core Components

### 1. Data Processing

### Data Augmentation Implementation

#### Rotation and Positioning (`Polluted_Images_Generation.py`)

```python
class CRRNWEP:
    def __init__(self, range1=(-30, -15), range2=(15, 30), size=(28, 28)):
        # Custom Random Rotation and Noisy With EdgePadding
        # Implements rotation, zoom, and positional adjustments
```

Key Features:

- Random rotation in specified ranges to simulate real-world variations
- Edge padding with calculated edge colors for seamless transformations
- Gaussian noise addition option for increased robustness
- Size normalization to maintain consistency

#### Oscillation Block Processing (`Image_oscillation.py`)

```python
class BatchWise_Oscillate:
    def __init__(self, images):
        # Processes batch of images to capture tilted features
```

Capabilities:

- Splits images into blocks (4×4 or 7×7 grids) for detailed feature analysis
- Applies two different ring-based reordering patterns to capture rotation variations
- Returns two oscillated versions of input for enhanced feature representation
- Designed specifically to handle tilted features without additional training data

### 2. Model Architectures

#### Standard CNN (`CNN.py`)

```python
class CNN(nn.Module):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims, **kwargs):
        # Standard CNN with ResNet18 integration
```

#### RDENET (`RDECNN.py`)

```python
class RDENet(CNN):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims, **kwargs):
        # Enhanced CNN with oscillation blocks
```

### 3. Training Scripts

#### Standard Training

```python
# Train_CNN.py - Single run training
# Train_CNN_with_Cross_Validation.py - Cross-validation training
```

#### RDENET Training

```python
# Train_RDECNN.py - Single run training
# Train_RDECNN_with_Cross_Validation.py - Cross-validation training
```

## Usage Examples

### Basic CNN Training

```python
from CNN import CNN
from Train_CNN_with_Cross_Validation import FashionMNIST_CNNClassifier

# Initialize classifier
classifier = FashionMNIST_CNNClassifier(
    batch_size=32,
    lr=1e-3,
    epochs=50,
    save_path='./outputs/cnn_results'
)

# Run cross-validation
classifier.cross_validate(
    training_loader=classifier.original_train_loader,
    training_type='original',
    k_folds=5
)
```

### Enhanced RDENET Training

```python
from RDECNN import RDENet
from Train_RDECNN_with_Cross_Validation import FashionMNIST_RDECNNClassifier

# Initialize classifier with oscillation-enhanced architecture
classifier = FashionMNIST_RDECNNClassifier(
    batch_size=32,
    lr=1e-3,
    epochs=50,
    save_path='./outputs/rdenet_results'
)

# Run cross-validation specifically on rotated data to test oscillation block effectiveness
classifier.cross_validate(
    training_loader=classifier.polluted_train_loader,
    training_type='polluted',
    k_folds=5
)
```

## Configuration Options

### Model Parameters

```python
params = {
    'in_size': (1, 28, 28),        # Input dimensions
    'out_classes': 10,             # Number of classes
    'channels': [16, 32, 64],      # CNN channel progression
    'pool_every': 2,               # Pooling frequency
    'hidden_dims': [128, 64],      # FC layer dimensions
    'conv_params': {
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'activation_type': 'relu',
    'pooling_type': 'max',
    'batchnorm': True,
    'dropout': 0.1
}
```

### Training Parameters

```python
training_config = {
    'batch_size': 32,              # Batch size
    'lr': 1e-3,                    # Learning rate
    'epochs': 50,                  # Maximum epochs
    'device': 'auto',              # Device selection
    'k_folds': 5                   # Cross-validation folds
}
```

## Data Loading

### Automatic Dataset Download

```python
# FashionMNIST automatically downloads if not present
train_dataset = datasets.FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=transform
)
```

### Data Transforms

```python
# Standard transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Polluted transform
polluted_transform = transforms.Compose([
    CRRNWEP(range1=(-30, -10), range2=(10, 30), size=(28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
```

## Output Structure

### Metrics Files

```text
outputs/
├── cnn_with_resnet_cross_validation/
│   ├── all_metrics.json              # Complete metrics
│   ├── cross_validation_metrics_*.json  # Per-scenario metrics
│   ├── mean_std_*.json               # Statistical summaries
│   └── training.log                  # Training logs
└── rdecnn_with_resnet_cross_validation/
    └── [same structure]
```

### Metrics Format

```json
{
    "cross_validation": {
        "fold_1": {
            "train_loss": [0.625, 0.428, ...],
            "train_accuracy": [77.2, 84.6, ...],
            "val_loss": [0.223],
            "val_accuracy": [92.0]
        }
    }
}
```
