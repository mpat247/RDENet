# Architecture Documentation

## Overview

This document provides detailed information about the CNN and RDENET architectures implemented in this project.

## Standard CNN Architecture

### Components

1. **Feature Extractor**: Convolutional layers with configurable parameters
2. **ResNet18 Integration**: Pre-trained ResNet18 backbone for enhanced feature extraction
3. **MLP Classifier**: Fully connected layers for final classification

### Configuration

```python
params = {
    'in_size': (1, 28, 28),      # FashionMNIST input dimensions
    'out_classes': 10,           # 10 clothing categories
    'channels': [16, 32, 64],    # Progressive channel expansion
    'pool_every': 2,             # Pooling frequency
    'hidden_dims': [128, 64],    # FC layer dimensions
    'conv_params': {
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'activation_type': 'relu',
    'pooling_type': 'max'
}
```

### Data Flow

```text
Input (1×28×28) → CNN Feature Extractor → ResNet18 → Flatten → MLP → Output (10)
```

## RDENET Architecture

### Key Innovation: Oscillation Block

The oscillation block is the core component that differentiates RDENET from standard CNN:

#### Components

1. **Main Path**: Standard convolution processing
2. **Bypass Path One**: Processes first oscillation pattern
3. **Bypass Path Two**: Processes second oscillation pattern
4. **Concatenation**: Combines all three paths

#### Oscillation Processing

The oscillation block utilizes the `BatchWise_Oscillate` function that:

- Splits input images into 4×4 or 7×7 blocks
- Applies two different ring-based reordering patterns
- Creates alternative representations of the same image

### Complete RDENET Flow

```text
Input → Mapping Layer → Oscillation Block → Feature Extractor →
ResNet18 → Classification Output
```

### Mapping Layer

Transforms flattened input before processing:

```python
class MappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn=nn.ReLU):
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation_fn()
```

## Comparison

| Component         | Standard CNN           | RDENET                        |
| ----------------- | ---------------------- | ----------------------------- |
| Input Processing  | Direct convolution     | Mapping layer + oscillation   |
| Feature Paths     | Single path            | Three parallel paths          |
| Rotation Handling | Data augmentation only | Built-in oscillation patterns |
| Complexity        | Lower                  | Higher                        |
| Parameters        | Fewer                  | More (due to multiple paths)  |

## Implementation Details

### Oscillation Block Forward Pass

```python
def forward(self, x: Tensor):
    main_output = self.main_path(x)
    one, two = BatchWise_Oscillate(x).get_result()
    shortcut_output_one = self.bypath_one(one)
    shortcut_output_two = self.bypath_two(two)
    out = torch.cat([main_output, shortcut_output_one, shortcut_output_two], dim=1)
    return out
```

### ResNet Integration

Both architectures integrate ResNet18:

- Modified first conv layer to accept appropriate channel count
- Removed maxpool layer to preserve spatial dimensions
- Adapted final layer for 10-class classification
- Used as feature extractor (removed original FC layer)
