# RDENET: Rotation-Detection-Enhanced CNN with Oscillation Block

A CNN architecture enhanced with oscillation blocks to improve robustness against image rotations on FashionMNIST dataset.

## Project Overview

This project implements and compares two CNN architectures:

- **Standard CNN**: Baseline CNN with ResNet18 integration
- **RDENET**: Enhanced CNN with oscillation blocks for rotation detection

The models are evaluated on normal, polluted (rotated), and mixed datasets using cross-validation.

## Architecture

### Standard CNN

- Convolutional layers with ReLU activation
- Max pooling and batch normalization
- ResNet18 integration for feature extraction
- Fully connected layers for classification

### RDENET (Enhanced Architecture)

- **Oscillation Block**: Core innovation that processes images through multiple pathways
- **Main Path**: Standard convolution processing
- **Bypass Paths**: Two parallel paths handling different oscillation patterns
- **Mapping Layer**: Transforms input features before processing
- **ResNet18 Integration**: For enhanced feature extraction

## Dataset

**FashionMNIST**: 28x28 grayscale images, 10 clothing categories

- Training set: 60,000 images
- Test set: 10,000 images
- **Data augmentation**: Random rotation pollution (±30° to ±10° and ±10° to ±30°)

## Results Summary

### Cross-Validation Performance (5-fold)

**CNN (Polluted Data)**:

- Training Accuracy: 91.61% ± 3.77%
- Validation Accuracy: 92.18% ± 0.18%
- Training Loss: 0.229 ± 0.106

**RDENET (Polluted Data)**:

- Training Accuracy: 85.48% ± 4.13%
- Validation Accuracy: 86.97% ± 0.15%
- Training Loss: 0.395 ± 0.110

## Repository Structure

```text
src/
├── CNN.py                              # Standard CNN implementation
├── RDECNN.py                          # RDENET with oscillation blocks
├── Image_oscillation.py               # Oscillation block implementation
├── Polluted_Images_Generation.py      # Data augmentation utilities
├── Train_CNN.py                       # CNN training script
├── Train_CNN_with_Cross_Validation.py # CNN cross-validation
├── Train_RDECNN.py                    # RDENET training script
└── Train_RDECNN_with_Cross_Validation.py # RDENET cross-validation

outputs/
├── cnn_with_resnet_cross_validation/   # CNN results
└── rdecnn_with_resnet_cross_validation/ # RDENET results

data/FashionMNIST/                      # Dataset storage
docs/                                   # Documentation
```

## Usage

### Training Standard CNN

```bash
cd src
python Train_CNN_with_Cross_Validation.py
```

### Training RDENET

```bash
cd src
python Train_RDECNN_with_Cross_Validation.py
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- scikit-learn
- tqdm
- PIL

## Key Features

- **Cross-validation**: 5-fold stratified cross-validation
- **Multiple training scenarios**: Normal, polluted, and mixed datasets
- **Comprehensive evaluation**: Training on different data types, testing on all scenarios
- **Detailed logging**: Training progress and metrics tracking
- **Model checkpointing**: Best model saving with early stopping

## Documentation

See `docs/` folder for detailed technical documentation and experimental results.
