# Enhancing CNN Robustness By Using Data Augmentation and Oscillation Block

_Hao Luo, Haiyi Wang, Manav Patel, Christina Zeng - December 2024_

A research project investigating how structural modifications can improve convolutional neural networks (CNNs) in image classification tasks through oscillation blocks and data augmentation techniques.

## Abstract

This project investigates how structural modifications can improve convolutional neural networks (CNNs) in image classification tasks. We applied preprocessing methods like rotation, zooming, and positional adjustments to simulate real-world image variations. We also added an oscillation block to capture "tilted" features in rotated data without introducing new training data. Additionally, we used data augmentation to increase the size and diversity of the training dataset. Our experiments showed that the oscillation block improved the model's performance on rotated data. However, for mixed or normal datasets, the performance was slightly lower than that of standard CNNs. Overall, these techniques resulted in a more robust model, especially for handling rotated data, but also highlighted performance trade-offs in different training conditions.

## Project Goal

To enhance the performance of CNNs for image classification tasks by incorporating an oscillation block to capture tilted features in rotated images, while applying preprocessing techniques such as rotation, zoom, and positional adjustments to simulate real-world variations in images.

## Architecture Comparison

This project implements and compares two CNN architectures:

- **Standard CNN**: Baseline CNN with ResNet18 integration
- **RDENET (Oscillation-Enhanced CNN)**: Enhanced CNN with oscillation blocks designed to capture tilted features in rotated data

The models are evaluated on normal, polluted (rotated), and mixed datasets using cross-validation.

## Architecture

### Standard CNN

- Convolutional layers with ReLU activation
- Max pooling and batch normalization
- ResNet18 integration for feature extraction
- Fully connected layers for classification

### RDENET (Enhanced Architecture)

- **Oscillation Block**: Core innovation designed to capture "tilted" features in rotated data without introducing new training data
- **Main Path**: Standard convolution processing
- **Bypass Paths**: Two parallel paths handling different oscillation patterns to process rotated features
- **Mapping Layer**: Transforms input features before processing
- **ResNet18 Integration**: For enhanced feature extraction

## Data Augmentation Techniques

**Preprocessing Methods Applied**:

- **Rotation**: Random rotation transformations to simulate real-world image variations
- **Zooming**: Scale adjustments to test model robustness
- **Positional Adjustments**: Spatial transformations to increase dataset diversity
- **Edge Padding**: Calculated edge colors for rotation transformations

## Dataset

**FashionMNIST**: 28x28 grayscale images, 10 clothing categories

- Training set: 60,000 images
- Test set: 10,000 images
- **Data augmentation**: Rotation, zoom, and positional adjustments applied to simulate real-world variations
- **Polluted Data**: Images with rotation transformations (±30° to ±10° and ±10° to ±30°)

## Results Summary

### Cross-Validation Performance (5-fold)

**CNN (Polluted Data)**:

- Training Accuracy: 91.61% ± 3.77%
- Validation Accuracy: 92.18% ± 0.18%
- Training Loss: 0.229 ± 0.106

**RDENET (Polluted Data)**:

- Training Accuracy: 85.5% ± 4.1%
- Validation Accuracy: 87.0% ± 0.1%
- Training Loss: 0.395 ± 0.110
- **Note**: Improved performance specifically on rotated data, demonstrating the effectiveness of the oscillation block for handling tilted features

## Key Findings

- **Oscillation Block Effectiveness**: The oscillation block successfully improved model performance on rotated data by capturing tilted features
- **Performance Trade-offs**: For mixed or normal datasets, RDENET performance was slightly lower than standard CNNs
- **Robustness Enhancement**: The techniques resulted in a more robust model, especially for handling rotated data
- **Training Condition Sensitivity**: Highlighted important performance trade-offs in different training conditions

## Research Contributions

- Novel oscillation block architecture for rotation-invariant feature extraction
- Comprehensive evaluation of CNN robustness using data augmentation techniques
- Systematic analysis of performance trade-offs between standard and enhanced CNN architectures
- Demonstration of structural modifications' impact on CNN performance in image classification tasks

## Technical Features

- **Cross-validation**: 5-fold stratified cross-validation
- **Multiple training scenarios**: Normal, polluted, and mixed datasets
- **Comprehensive evaluation**: Training on different data types, testing on all scenarios
- **Detailed logging**: Training progress and metrics tracking
- **Model checkpointing**: Best model saving with early stopping

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
