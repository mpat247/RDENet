# Train_RDENet.py

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from tqdm import tqdm
from RDECNN import RDENet  # Ensure RDECNN.py is in the same directory or properly referenced
from Polluted_Images_Generation import CRRNWEP  # Ensure this module is correctly implemented
from sklearn.model_selection import StratifiedKFold  # For cross-validation
import os
import numpy as np
import json
from datetime import datetime

# --- Additional Imports for Enhanced Functionality ---
import logging
from collections import defaultdict
from functools import partial  # Import partial for scheduler initialization

# --- Define Activation and Pooling Dictionaries ---
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Default keyword arguments to pass to activation class constructors
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class FashionMNIST_RDECNNClassifier:
    def __init__(self, batch_size=32, lr=1e-3, epochs=50, device=None, save_path='./outputs/rdecnn_with_resnet'):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        if device:
            self.device = device
        else:
            # Safely check for MPS support
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()  # Clears the MPS cache
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        self.save_path = save_path

        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        # --- Initialize Metrics Dictionary ---
        self.metrics = defaultdict(dict)

        # --- Initialize Logger ---
        self.logger = self.setup_logger()

        # Data preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Anti-normalize for visualization
        self.inv_normalize = transforms.Normalize(
            mean=(-0.5 / 0.5,), std=(1 / 0.5,)
        )

        # Define the transform with normalization and pollution
        self.polluting_transform = transforms.Compose([
            CRRNWEP(range1=(-30, -10), range2=(10, 30), size=(28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Load and split data
        self.load_and_split_data()

        # Initialize model
        self.model = self.initialize_model()

        # Loss function, optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.logger.info(f'Using device: {self.device}')
        print(f'Using device: {self.device}')

    # --- Setup Logger ---
    def setup_logger(self):
        logger = logging.getLogger(self.save_path)
        logger.setLevel(logging.INFO)
        # Prevent adding multiple handlers if the logger already has handlers
        if not logger.handlers:
            # Create handlers
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler(os.path.join(self.save_path, 'training.log'))
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.INFO)
            # Create formatters and add to handlers
            c_format = logging.Formatter('%(asctime)s - %(message)s')
            f_format = logging.Formatter('%(asctime)s - %(message)s')
            c_handler.setFormatter(c_format)
            f_handler.setFormatter(f_format)
            # Add handlers to the logger
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)
        return logger

    def set_path(self, path):
        self.save_path = path
        os.makedirs(self.save_path, exist_ok=True)

    def load_and_split_data(self):
        """
        Load the original and polluted datasets, then create DataLoaders.
        """
        # Load original datasets
        original_train_full = datasets.FashionMNIST(root='../data', train=True, download=True, transform=self.transform)
        original_test_full = datasets.FashionMNIST(root='../data', train=False, download=True, transform=self.transform)

        # Create DataLoaders for original datasets
        self.original_train_loader = DataLoader(original_train_full, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.original_test_loader = DataLoader(original_test_full, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Load polluted datasets (apply polluted_transform)
        polluted_train_full = datasets.FashionMNIST(root='../data', train=True, download=True, transform=self.polluting_transform)
        polluted_test_full = datasets.FashionMNIST(root='../data', train=False, download=True, transform=self.polluting_transform)

        # Create DataLoaders for polluted datasets
        self.polluted_train_loader = DataLoader(polluted_train_full, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.polluted_test_loader = DataLoader(polluted_test_full, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Create mixed datasets by concatenating original and polluted train sets
        mixed_train = ConcatDataset([original_train_full, polluted_train_full])
        mixed_test = ConcatDataset([original_test_full, polluted_test_full])

        # Create DataLoaders for mixed datasets
        self.mixed_train_loader = DataLoader(mixed_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.mixed_test_loader = DataLoader(mixed_test, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.logger.info("Data loaded successfully for all training scenarios.")
        print("Data loaded successfully for all training scenarios.")

    def initialize_model(self):
        self.params = dict(
            in_size=(1, 28, 28),  # Input shape of FashionMNIST
            out_classes=10,  # FashionMNIST has 10 classes
            channels=[16, 32, 64],  # Channels in each convolutional layer
            pool_every=2,  # Apply pooling after every 2 layers
            hidden_dims=[128, 64],  # Fully connected layers
            conv_params=dict(kernel_size=3, stride=1, padding=1),
            activation_type='relu',
            activation_params=dict(),
            pooling_type='max',
            pooling_params=dict(kernel_size=2, stride=2),
            batchnorm=True, 
            dropout=0.1,
            bottleneck=False
        )
        self.logger.info("Initializing RDENet with ResNet integration")
        print("Initializing RDENet with ResNet integration")
        model = RDENet(**self.params).to(self.device)
        self.logger.info("RDENet initialized successfully")
        print("RDENet initialized successfully")
        return model

    # --- Save Metrics ---
    def save_metrics(self, metrics_path):
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved at {metrics_path}")
        print(f"Metrics saved at {metrics_path}")

    # --- Save Model ---
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved at {path}")
        print(f"Model saved at {path}")

    # --- Load Model ---
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.logger.info(f"Loaded model from {path}")
        print(f"Loaded model from {path}")

    # --- Recursive Get Targets ---
    def get_targets(self, dataset):
        """
        Recursively extract targets from a dataset, handling Subset and ConcatDataset.

        :param dataset: Dataset object (e.g., Subset, ConcatDataset, etc.).
        :return: Numpy array of targets.
        """
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif isinstance(dataset, Subset):
            return self.get_targets(dataset.dataset)[dataset.indices]
        elif isinstance(dataset, ConcatDataset):
            targets = []
            for subset in dataset.datasets:
                targets.append(self.get_targets(subset))
            return np.concatenate(targets)
        else:
            raise AttributeError("Unsupported dataset type for target extraction.")

    # --- Cross-Validation ---
    def cross_validate(self, training_loader, training_type='original', k_folds=5, scheduler=None):
        """
        Perform k-fold cross-validation on the given training dataset.

        :param training_loader: DataLoader for the training dataset.
        :param training_type: Type of training ('original', 'polluted', 'mixed').
        :param k_folds: Number of folds for cross-validation.
        :param scheduler: Learning rate scheduler factory (callable that takes optimizer and returns scheduler).
        :return: Dictionary containing validation loss and accuracy for each fold.
        """
        self.logger.info(f"=== Starting {k_folds}-Fold Cross-Validation for {training_type.capitalize()} Training ===")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Extract targets from the dataset
        y = self.get_targets(training_loader.dataset)

        X = np.arange(len(y))  # Sample indices

        fold_results = {}

        for fold, (train_indices, val_indices) in enumerate(skf.split(X, y)):
            # Check if this fold has already been completed
            fold_metrics_key = f"{training_type}_fold_{fold + 1}"
            if fold_metrics_key in self.metrics["cross_validation"]:
                self.logger.info(f"--- Fold {fold + 1} for {training_type.capitalize()} Training already completed. Skipping... ---")
                print(f"--- Fold {fold + 1} for {training_type.capitalize()} Training already completed. Skipping... ---")
                continue  # Skip this fold

            self.logger.info(f"--- Processing Fold {fold + 1} ---")
            print(f"--- Processing Fold {fold + 1} ---")
            val_loss, val_accuracy = self.train_fold(train_indices, val_indices, fold, training_type, scheduler)
            fold_results[fold + 1] = {'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy}

        self.logger.info(f"=== Cross-Validation Results for {training_type.capitalize()} Training ===")
        for fold in fold_results:
            self.logger.info(f"Fold {fold}: Loss={fold_results[fold]['Validation Loss']:.4f}, Accuracy={fold_results[fold]['Validation Accuracy']:.2f}%")
            print(f"Fold {fold}: Loss={fold_results[fold]['Validation Loss']:.4f}, Accuracy={fold_results[fold]['Validation Accuracy']:.2f}%")

        # Save cross-validation metrics
        metrics_file = os.path.join(self.save_path, f'cross_validation_metrics_{training_type}.json')
        self.save_metrics(metrics_file)

        return fold_results

    # --- Train Fold ---
    def train_fold(self, train_indices, val_indices, fold, training_type, scheduler=None):
        """
        Train the model on a specific fold.

        :param train_indices: Indices for training data.
        :param val_indices: Indices for validation data.
        :param fold: Current fold number.
        :param training_type: Type of training ('original', 'polluted', 'mixed').
        :param scheduler: Learning rate scheduler factory (callable that takes optimizer and returns scheduler).
        :return: Tuple of (validation loss, validation accuracy)
        """
        self.logger.info(f"=== Starting {training_type.capitalize()} Fold {fold + 1} ===")
        print(f"=== Starting {training_type.capitalize()} Fold {fold + 1} ===")
        # Create data subsets
        train_subset = Subset(self.get_training_loader(training_type, split='train').dataset, train_indices)
        val_subset = Subset(self.get_training_loader(training_type, split='train').dataset, val_indices)  # Use train dataset for cross-validation

        # Data loaders for this fold
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Initialize a fresh model for each fold
        self.logger.info(f"Initializing model for {training_type.capitalize()} Fold {fold + 1}")
        print(f"Initializing model for {training_type.capitalize()} Fold {fold + 1}")
        model = RDENet(**self.params).to(self.device)
        model.train()

        # Define optimizer and loss for this fold
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Define scheduler if provided
        if scheduler:
            try:
                scheduler_instance = scheduler(optimizer)
                self.logger.info(f"Scheduler {scheduler_instance.__class__.__name__} initialized for Fold {fold + 1}")
                print(f"Scheduler {scheduler_instance.__class__.__name__} initialized for Fold {fold + 1}")
            except Exception as e:
                self.logger.error(f"Error initializing scheduler for Fold {fold + 1}: {e}")
                print(f"Error initializing scheduler for Fold {fold + 1}: {e}")
                scheduler_instance = None
        else:
            scheduler_instance = None

        best_accuracy = 0.0
        patience = 5  # For early stopping
        counter = 0

        # Initialize metrics for this fold
        self.metrics["cross_validation"][f"{training_type}_fold_{fold + 1}"] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(self.epochs):
            self.logger.info(f"{training_type.capitalize()} Fold {fold + 1}, Epoch {epoch + 1}/{self.epochs}")
            print(f"{training_type.capitalize()} Fold {fold + 1}, Epoch {epoch + 1}/{self.epochs}")
            running_loss = 0.0
            correct, total = 0, 0

            try:
                with tqdm(train_loader, desc=f'{training_type.capitalize()} Fold {fold + 1}, Epoch {epoch + 1}/{self.epochs}') as pbar:
                    for images, labels in pbar:
                        images, labels = images.to(self.device), labels.to(self.device)

                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        # Compute loss and accuracy
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        pbar.set_postfix({'Loss': running_loss / (total / self.batch_size),
                                          'Accuracy': 100. * correct / total})

                avg_loss = running_loss / len(train_loader)
                accuracy = 100 * correct / total
                self.metrics["cross_validation"][f"{training_type}_fold_{fold + 1}"]["train_loss"].append(avg_loss)
                self.metrics["cross_validation"][f"{training_type}_fold_{fold + 1}"]["train_accuracy"].append(accuracy)
                self.logger.info(f"{training_type.capitalize()} Fold {fold + 1}, Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                print(f"{training_type.capitalize()} Fold {fold + 1}, Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

                # Step the scheduler if defined
                if scheduler_instance:
                    try:
                        scheduler_instance.step()
                        self.logger.info(f"Scheduler stepped for {training_type.capitalize()} Fold {fold + 1}")
                        print(f"Scheduler stepped for {training_type.capitalize()} Fold {fold + 1}")
                    except Exception as e:
                        self.logger.error(f"Error stepping scheduler for Fold {fold + 1}: {e}")
                        print(f"Error stepping scheduler for Fold {fold + 1}: {e}")

                # Save checkpoint
                save_checkpoints = os.path.join(self.save_path, f"{training_type}_fold_{fold + 1}_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), save_checkpoints)
                self.logger.info(f"Checkpoint saved at {save_checkpoints}")
                print(f"Checkpoint saved at {save_checkpoints}")

                # Early Stopping Check
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    counter = 0
                    # Save the best model for this fold
                    best_model_path = os.path.join(self.save_path, f"{training_type}_fold_{fold + 1}_best.pt")
                    torch.save(model.state_dict(), best_model_path)
                    self.logger.info(f"New best model for {training_type.capitalize()} Fold {fold + 1} saved at {best_model_path}")
                    print(f"New best model for {training_type.capitalize()} Fold {fold + 1} saved at {best_model_path}")
                else:
                    counter += 1
                    self.logger.info(f"No improvement for {training_type.capitalize()} Fold {fold + 1}, Counter: {counter}/{patience}")
                    print(f"No improvement for {training_type.capitalize()} Fold {fold + 1}, Counter: {counter}/{patience}")
                    if counter >= patience:
                        self.logger.info(f"Early stopping triggered for {training_type.capitalize()} Fold {fold + 1} at epoch {epoch + 1}")
                        print(f"Early stopping triggered for {training_type.capitalize()} Fold {fold + 1} at epoch {epoch + 1}")
                        break

            except Exception as e:
                self.logger.error(f"Error during training in Fold {fold + 1}, Epoch {epoch + 1}: {e}")
                print(f"Error during training in Fold {fold + 1}, Epoch {epoch + 1}: {e}")
                break  # Optionally, you can continue to the next epoch or fold

        # Load the best model for validation
        try:
            self.logger.info(f"Loading best model for {training_type.capitalize()} Fold {fold + 1} from {best_model_path}")
            print(f"Loading best model for {training_type.capitalize()} Fold {fold + 1} from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
        except Exception as e:
            self.logger.error(f"Error loading best model for Fold {fold + 1}: {e}")
            print(f"Error loading best model for Fold {fold + 1}: {e}")
            return float('inf'), 0.0  # Return worst metrics if loading fails

        val_loss = 0.0
        val_correct, val_total = 0, 0

        try:
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f'{training_type.capitalize()} Fold {fold + 1} Validation', leave=False):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            self.metrics["cross_validation"][f"{training_type}_fold_{fold + 1}"]["val_loss"].append(avg_val_loss)
            self.metrics["cross_validation"][f"{training_type}_fold_{fold + 1}"]["val_accuracy"].append(val_accuracy)
            self.logger.info(f"{training_type.capitalize()} Fold {fold + 1} Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            print(f"{training_type.capitalize()} Fold {fold + 1} Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        except Exception as e:
            self.logger.error(f"Error during validation for Fold {fold + 1}: {e}")
            print(f"Error during validation for Fold {fold + 1}: {e}")
            avg_val_loss = float('inf')
            val_accuracy = 0.0

        return avg_val_loss, val_accuracy

    # --- Helper Method to Get Training Loader ---
    def get_training_loader(self, training_type='original', split='train'):
        """
        Retrieve the appropriate training DataLoader based on the training type and split.

        :param training_type: Type of training ('original', 'polluted', 'mixed').
        :param split: Split type ('train', 'test').
        :return: DataLoader corresponding to the training type and split.
        """
        if training_type == 'original':
            if split == 'train':
                return self.original_train_loader
            elif split == 'test':
                return self.original_test_loader
        elif training_type == 'polluted':
            if split == 'train':
                return self.polluted_train_loader
            elif split == 'test':
                return self.polluted_test_loader
        elif training_type == 'mixed':
            if split == 'train':
                return self.mixed_train_loader
            elif split == 'test':
                return self.mixed_test_loader
        else:
            raise ValueError(f"Unsupported training type: {training_type}")

    # --- Train and Validate Method ---
    def train_and_validate(self, training_type='original', k_folds=5, scheduler=None):
        """
        Perform cross-validation for a specific training scenario.

        :param training_type: Type of training ('original', 'polluted', 'mixed').
        :param k_folds: Number of folds for cross-validation.
        :param scheduler: Learning rate scheduler factory (callable that takes optimizer and returns scheduler).
        """
        training_loader = self.get_training_loader(training_type, split='train')
        self.cross_validate(training_loader, training_type, k_folds, scheduler)

    # --- Testing Methods ---
    def test_model(self, test_loader, test_type='original', training_type='original'):
        """
        Test the model on the given test dataset.

        :param test_loader: DataLoader for the test dataset.
        :param test_type: Type of test ('original', 'polluted').
        :param training_type: Type of training the model underwent ('original', 'polluted', 'mixed').
        """
        self.logger.info(f"=== Testing on {test_type.capitalize()} Test Data after {training_type.capitalize()} Training ===")
        print(f"=== Testing on {test_type.capitalize()} Test Data after {training_type.capitalize()} Training ===")
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        metrics_key = f"test_after_{training_type}_{test_type}"
        self.metrics[metrics_key] = {"loss": 0.0, "accuracy": 0.0}

        try:
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc=f"Testing {test_type.capitalize()} Data after {training_type.capitalize()} Training", leave=False):
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(test_loader)
            accuracy = 100 * correct / total
            self.metrics[metrics_key]["loss"] = avg_loss
            self.metrics[metrics_key]["accuracy"] = accuracy
            self.logger.info(f"{test_type.capitalize()} Test after {training_type.capitalize()} Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"{test_type.capitalize()} Test after {training_type.capitalize()} Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        except Exception as e:
            self.logger.error(f"Error during testing on {test_type.capitalize()} data after {training_type.capitalize()} training: {e}")
            print(f"Error during testing on {test_type.capitalize()} data after {training_type.capitalize()} training: {e}")
            self.metrics[metrics_key]["loss"] = float('inf')
            self.metrics[metrics_key]["accuracy"] = 0.0

    # --- Calculate Cross-Validation Statistics ---
    def calculate_cv_statistics(self, training_type='original'):
        """
        Calculate mean and standard deviation of cross-validation metrics.

        :param training_type: Type of training ('original', 'polluted', 'mixed').
        """
        fold_keys = [key for key in self.metrics["cross_validation"] if key.startswith(training_type)]
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for key in fold_keys:
            train_losses.extend(self.metrics["cross_validation"][key]["train_loss"])
            train_accuracies.extend(self.metrics["cross_validation"][key]["train_accuracy"])
            val_losses.extend(self.metrics["cross_validation"][key]["val_loss"])
            val_accuracies.extend(self.metrics["cross_validation"][key]["val_accuracy"])

        # Calculate statistics
        stats = {}
        if train_losses:
            stats["train_loss_mean"] = float(np.mean(train_losses))
            stats["train_loss_std"] = float(np.std(train_losses))
        if train_accuracies:
            stats["train_accuracy_mean"] = float(np.mean(train_accuracies))
            stats["train_accuracy_std"] = float(np.std(train_accuracies))
        if val_losses:
            stats["val_loss_mean"] = float(np.mean(val_losses))
            stats["val_loss_std"] = float(np.std(val_losses))
        if val_accuracies:
            stats["val_accuracy_mean"] = float(np.mean(val_accuracies))
            stats["val_accuracy_std"] = float(np.std(val_accuracies))

        # Save statistics
        stats_file = os.path.join(self.save_path, f'cross_validation_stats_{training_type}.json')
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            self.logger.info(f"Cross-validation statistics saved at {stats_file}")
            print(f"Cross-validation statistics saved at {stats_file}")
        except Exception as e:
            self.logger.error(f"Error saving cross-validation statistics for {training_type.capitalize()} training: {e}")
            print(f"Error saving cross-validation statistics for {training_type.capitalize()} training: {e}")

        # Log statistics
        self.logger.info(f"=== Cross-Validation Statistics for {training_type.capitalize()} Training ===")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value:.4f}")
            print(f"{key}: {value:.4f}")

    # --- Compute and Save Mean and Std ---
    def compute_and_save_mean_std(self):
        """
        Manually compute mean and standard deviation for each training scenario
        based on the cross-validation metrics JSON files. Save the results as
        mean_std_{training_type}.json.
        """
        training_types = ['original', 'polluted', 'mixed']
        for training_type in training_types:
            mean_std_file = os.path.join(self.save_path, f'mean_std_{training_type}.json')
            cross_val_metrics_file = os.path.join(self.save_path, f'cross_validation_metrics_{training_type}.json')

            # Check if mean_std file already exists
            if os.path.exists(mean_std_file):
                self.logger.info(f"Mean and Std for {training_type.capitalize()} training already computed. Skipping...")
                print(f"Mean and Std for {training_type.capitalize()} training already computed. Skipping...")
                continue

            # Check if cross-validation metrics file exists
            if not os.path.exists(cross_val_metrics_file):
                self.logger.warning(f"Cross-validation metrics file for {training_type.capitalize()} training not found. Cannot compute mean and std.")
                print(f"Cross-validation metrics file for {training_type.capitalize()} training not found. Cannot compute mean and std.")
                continue

            # Load cross-validation metrics
            try:
                with open(cross_val_metrics_file, 'r') as f:
                    cv_metrics = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cross-validation metrics for {training_type.capitalize()} training: {e}")
                print(f"Error loading cross-validation metrics for {training_type.capitalize()} training: {e}")
                continue

            # Check if 'cross_validation' key exists
            if "cross_validation" not in cv_metrics:
                self.logger.error(f"'cross_validation' key not found in {cross_val_metrics_file}. Cannot compute mean and std.")
                print(f"'cross_validation' key not found in {cross_val_metrics_file}. Cannot compute mean and std.")
                continue

            folds = cv_metrics["cross_validation"]

            # Initialize lists to store metrics across folds
            train_loss_list = []
            train_accuracy_list = []
            val_loss_list = []
            val_accuracy_list = []

            # Aggregate metrics from each fold
            for fold_key, metrics in folds.items():
                train_loss_list.extend(metrics.get("train_loss", []))
                train_accuracy_list.extend(metrics.get("train_accuracy", []))
                val_loss_list.extend(metrics.get("val_loss", []))
                val_accuracy_list.extend(metrics.get("val_accuracy", []))

            # Compute mean and std
            mean_std = {}
            if train_loss_list:
                mean_std["train_loss_mean"] = float(np.mean(train_loss_list))
                mean_std["train_loss_std"] = float(np.std(train_loss_list))
            if train_accuracy_list:
                mean_std["train_accuracy_mean"] = float(np.mean(train_accuracy_list))
                mean_std["train_accuracy_std"] = float(np.std(train_accuracy_list))
            if val_loss_list:
                mean_std["val_loss_mean"] = float(np.mean(val_loss_list))
                mean_std["val_loss_std"] = float(np.std(val_loss_list))
            if val_accuracy_list:
                mean_std["val_accuracy_mean"] = float(np.mean(val_accuracy_list))
                mean_std["val_accuracy_std"] = float(np.std(val_accuracy_list))

            # Save mean and std to JSON file
            try:
                with open(mean_std_file, 'w') as f:
                    json.dump(mean_std, f, indent=4)
                self.logger.info(f"Mean and Std for {training_type.capitalize()} training saved at {mean_std_file}")
                print(f"Mean and Std for {training_type.capitalize()} training saved at {mean_std_file}")
            except Exception as e:
                self.logger.error(f"Error saving mean and std for {training_type.capitalize()} training: {e}")
                print(f"Error saving mean and std for {training_type.capitalize()} training: {e}")
                continue

            # Log and print the mean and std
            self.logger.info(f"=== Mean and Std for {training_type.capitalize()} Training ===")
            print(f"=== Mean and Std for {training_type.capitalize()} Training ===")
            for key, value in mean_std.items():
                self.logger.info(f"{key}: {value:.4f}")
                print(f"{key}: {value:.4f}")

    # --- Check Cross-Validation Completion ---
    def is_cross_validation_complete(self):
        """
        Check if cross-validation has been completed for all training scenarios.

        :return: Boolean indicating completion.
        """
        training_types = ['original', 'polluted', 'mixed']
        for training_type in training_types:
            metrics_file = os.path.join(self.save_path, f'cross_validation_metrics_{training_type}.json')
            if not os.path.exists(metrics_file):
                return False
        return True

    # --- Check Normal Training Completion ---
    def is_normal_training_complete(self):
        """
        Check if normal training has been completed for all training scenarios.

        :return: Boolean indicating completion.
        """
        training_types = ['original', 'polluted', 'mixed']
        for training_type in training_types:
            best_model_path = os.path.join(self.save_path, f"{training_type}_standard_best.pt")
            if not os.path.exists(best_model_path):
                return False
        return True

    # --- Final Metrics Compilation ---
    def compile_final_metrics(self):
        """
        Compile all metrics into a final_all_complete_metrics.json file.
        """
        final_metrics = self.metrics.copy()
        final_metrics_file = os.path.join(self.save_path, 'final_all_complete_metrics.json')
        with open(final_metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        self.logger.info(f"Final metrics compiled and saved at {final_metrics_file}")
        print(f"Final metrics compiled and saved at {final_metrics_file}")

    # --- Train with Cross-Validation and Standard Training ---
    def train_with_cross_validation_and_standard(self, k_folds=5):
        """
        Perform cross-validation and standard training/testing based on completion.

        :param k_folds: Number of folds for cross-validation.
        """
        training_types = ['original', 'polluted', 'mixed']
        all_cross_val_done = True

        for training_type in training_types:
            # Check if cross-validation for this training type is done
            metrics_file = os.path.join(self.save_path, f'cross_validation_metrics_{training_type}.json')
            if not os.path.exists(metrics_file):
                all_cross_val_done = False
                self.logger.info(f"\n=== Starting Cross-Validation for {training_type.capitalize()} Training ===")
                print(f"\n=== Starting Cross-Validation for {training_type.capitalize()} Training ===")
                # Define a learning rate scheduler (optional)
                # Example: StepLR with step_size=7 and gamma=0.1
                scheduler = partial(optim.lr_scheduler.StepLR, step_size=7, gamma=0.1)  # Initialize StepLR with required params
                self.logger.info(f"Using Scheduler: StepLR for {training_type.capitalize()} Training with step_size=7, gamma=0.1")
                print(f"Using Scheduler: StepLR for {training_type.capitalize()} Training with step_size=7, gamma=0.1")
                self.train_and_validate(training_type, k_folds, scheduler)
                # Calculate and save statistics
                self.calculate_cv_statistics(training_type)
            else:
                self.logger.info(f"=== Cross-Validation for {training_type.capitalize()} Training already completed. Skipping... ===")
                print(f"=== Cross-Validation for {training_type.capitalize()} Training already completed. Skipping... ===")

        # After cross-validation, compute mean and std manually
        self.compute_and_save_mean_std()

        # Check if mean and std files exist for all training types
        mean_std_complete = True
        for training_type in training_types:
            mean_std_file = os.path.join(self.save_path, f'mean_std_{training_type}.json')
            if not os.path.exists(mean_std_file):
                mean_std_complete = False
                self.logger.warning(f"Mean and Std file for {training_type.capitalize()} training not found. Cannot proceed to Standard Training.")
                print(f"Mean and Std file for {training_type.capitalize()} training not found. Cannot proceed to Standard Training.")

        if mean_std_complete and self.is_cross_validation_complete() and not self.is_normal_training_complete():
            # Proceed to standard training and testing
            self.logger.info("\n=== Starting Standard Training for All Scenarios ===")
            print("\n=== Starting Standard Training for All Scenarios ===")
            for training_type in training_types:
                self.train_normal(training_type, scheduler=partial(optim.lr_scheduler.StepLR, step_size=7, gamma=0.1))
                # After training, test on the original and polluted test sets
                original_test_loader = self.get_training_loader(training_type='original', split='test')
                polluted_test_loader = self.get_training_loader(training_type='polluted', split='test')
                self.test_model(original_test_loader, test_type='original', training_type=training_type)
                self.test_model(polluted_test_loader, test_type='polluted', training_type=training_type)
        elif self.is_normal_training_complete():
            self.logger.info("=== Standard Training already completed for all scenarios. Skipping... ===")
            print("=== Standard Training already completed for all scenarios. Skipping... ===")
        else:
            self.logger.info("=== Mean and Std not fully computed. Cannot proceed to Standard Training. ===")
            print("=== Mean and Std not fully computed. Cannot proceed to Standard Training. ===")

        # After all training, compile final metrics
        if self.is_cross_validation_complete() and self.is_normal_training_complete():
            self.compile_final_metrics()

    # --- Calculate Cross-Validation Statistics for All Training Types ---
    def calculate_all_cv_statistics(self):
        """
        Calculate and log mean and standard deviation for all training scenarios' cross-validation results.
        """
        training_types = ['original', 'polluted', 'mixed']
        for training_type in training_types:
            metrics_file = os.path.join(self.save_path, f'cross_validation_metrics_{training_type}.json')
            if os.path.exists(metrics_file):
                self.calculate_cv_statistics(training_type)
            else:
                self.logger.warning(f"Cross-validation metrics file for {training_type.capitalize()} Training not found.")
                print(f"Cross-validation metrics file for {training_type.capitalize()} Training not found.")

    # --- Save All Metrics ---
    def save_all_metrics(self):
        """
        Save all recorded metrics to a JSON file.
        """
        metrics_file = os.path.join(self.save_path, 'all_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"All metrics saved at {metrics_file}")
        print(f"All metrics saved at {metrics_file}")

    # --- Train Normal ---
    def train_normal(self, training_type='original', scheduler=None):
        """
        Perform standard training (not cross-validation) using the training and validation sets.

        :param training_type: Type of training ('original', 'polluted', 'mixed').
        :param scheduler: Learning rate scheduler factory (callable that takes optimizer and returns scheduler).
        """
        self.logger.info(f"=== Starting Standard Training for {training_type.capitalize()} ===")
        print(f"=== Starting Standard Training for {training_type.capitalize()} ===")

        # Split the training data into training and validation subsets
        full_train_loader = self.get_training_loader(training_type, split='train')
        dataset = full_train_loader.dataset
        total = len(dataset)
        val_size = int(0.2 * total)  # 20% for validation
        train_size = total - val_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Create DataLoaders for standard training
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Initialize a fresh model
        self.logger.info(f"Initializing model for Standard Training: {training_type.capitalize()}")
        print(f"Initializing model for Standard Training: {training_type.capitalize()}")
        model = RDENet(**self.params).to(self.device)
        model.train()

        # Define optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Define scheduler if provided
        if scheduler:
            try:
                scheduler_instance = scheduler(optimizer)
                self.logger.info(f"Scheduler {scheduler_instance.__class__.__name__} initialized for Standard Training")
                print(f"Scheduler {scheduler_instance.__class__.__name__} initialized for Standard Training")
            except Exception as e:
                self.logger.error(f"Error initializing scheduler for Standard Training: {e}")
                print(f"Error initializing scheduler for Standard Training: {e}")
                scheduler_instance = None
        else:
            scheduler_instance = None

        best_accuracy = 0.0
        patience = 5  # For early stopping
        counter = 0

        # Initialize metrics for standard training
        self.metrics["standard_training"] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(self.epochs):
            self.logger.info(f"Standard Training, Epoch {epoch + 1}/{self.epochs}")
            print(f"Standard Training, Epoch {epoch + 1}/{self.epochs}")
            running_loss = 0.0
            correct, total = 0, 0

            try:
                with tqdm(train_loader, desc=f'Standard Training, Epoch {epoch + 1}/{self.epochs}') as pbar:
                    for images, labels in pbar:
                        images, labels = images.to(self.device), labels.to(self.device)

                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        # Compute loss and accuracy
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        pbar.set_postfix({'Loss': running_loss / (total / self.batch_size),
                                          'Accuracy': 100. * correct / total})

            except Exception as e:
                self.logger.error(f"Error during Standard Training, Epoch {epoch + 1}: {e}")
                print(f"Error during Standard Training, Epoch {epoch + 1}: {e}")
                break  # Optionally, you can continue to the next epoch

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            self.metrics["standard_training"]["train_loss"].append(avg_loss)
            self.metrics["standard_training"]["train_accuracy"].append(accuracy)
            self.logger.info(f"Standard Training, Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Standard Training, Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Step the scheduler if defined
            if scheduler_instance:
                try:
                    scheduler_instance.step()
                    self.logger.info(f"Scheduler stepped for Standard Training")
                    print(f"Scheduler stepped for Standard Training")
                except Exception as e:
                    self.logger.error(f"Error stepping scheduler for Standard Training: {e}")
                    print(f"Error stepping scheduler for Standard Training: {e}")

            # Save checkpoint
            save_checkpoints = os.path.join(self.save_path, f"{training_type}_standard_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), save_checkpoints)
            self.logger.info(f"Checkpoint saved at {save_checkpoints}")
            print(f"Checkpoint saved at {save_checkpoints}")

            # Early Stopping Check
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                counter = 0
                # Save the best model for standard training
                best_model_path = os.path.join(self.save_path, f"{training_type}_standard_best.pt")
                torch.save(model.state_dict(), best_model_path)
                self.logger.info(f"New best model for Standard Training saved at {best_model_path}")
                print(f"New best model for Standard Training saved at {best_model_path}")
            else:
                counter += 1
                self.logger.info(f"No improvement for Standard Training, Counter: {counter}/{patience}")
                print(f"No improvement for Standard Training, Counter: {counter}/{patience}")
                if counter >= patience:
                    self.logger.info(f"Early stopping triggered for Standard Training at epoch {epoch + 1}")
                    print(f"Early stopping triggered for Standard Training at epoch {epoch + 1}")
                    break

            # Validation after each epoch
            val_loss = 0.0
            val_correct, val_total = 0, 0

            try:
                with torch.no_grad():
                    for images, labels in tqdm(val_loader, desc='Standard Training Validation', leave=False):
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                self.metrics["standard_training"]["val_loss"].append(avg_val_loss)
                self.metrics["standard_training"]["val_accuracy"].append(val_accuracy)
                self.logger.info(f"Standard Training Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                print(f"Standard Training Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            except Exception as e:
                self.logger.error(f"Error during Standard Training Validation: {e}")
                print(f"Error during Standard Training Validation: {e}")
                avg_val_loss = float('inf')
                val_accuracy = 0.0

        # Load the best model for validation (if needed)
        try:
            self.logger.info(f"Loading best model for Standard Training from {best_model_path}")
            print(f"Loading best model for Standard Training from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
        except Exception as e:
            self.logger.error(f"Error loading best model for Standard Training: {e}")
            print(f"Error loading best model for Standard Training: {e}")
            return float('inf'), 0.0  # Return worst metrics if loading fails

        return avg_val_loss, val_accuracy


def get_next_run_folder(outputs_dir='./outputs', base_name='rdecnn_with_resnet'):
    """
    Determines the next run folder name by incrementing the existing run folders.
    If an incomplete run is found, it resumes training in that folder.

    :param outputs_dir: Directory where run folders are stored.
    :param base_name: Base name for the run folders.
    :return: Tuple containing the run folder path and a boolean indicating if it's a resumed run.
    """
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    existing_folders = [folder for folder in os.listdir(outputs_dir) if folder.startswith(base_name)]
    run_numbers = [int(folder.split('_')[-1]) for folder in existing_folders if folder.split('_')[-1].isdigit()]
    run_numbers_sorted = sorted(run_numbers, reverse=True)
    
    # Define training scenarios
    training_types = ['original', 'polluted', 'mixed']

    # Check for incomplete runs starting from the latest
    for run_num in run_numbers_sorted:
        run_folder_name = f"{base_name}_{run_num}"
        run_folder_path = os.path.join(outputs_dir, run_folder_name)
        final_metrics_file = os.path.join(run_folder_path, 'final_all_complete_metrics.json')
        if os.path.exists(final_metrics_file):
            continue  # Run is complete, skip
        # Check if cross-validation metrics exist
        cross_val_complete = True
        for training_type in training_types:
            metrics_file = os.path.join(run_folder_path, f'cross_validation_metrics_{training_type}.json')
            if not os.path.exists(metrics_file):
                cross_val_complete = False
                break
        if cross_val_complete:
            # Check if standard training is complete
            standard_training_complete = True
            for training_type in training_types:
                best_model_path = os.path.join(run_folder_path, f"{training_type}_standard_best.pt")
                if not os.path.exists(best_model_path):
                    standard_training_complete = False
                    break
            if not standard_training_complete:
                # Incomplete run, resume standard training
                print(f"Found incomplete run folder: {run_folder_path}. Resuming standard training in this folder.")
                return run_folder_path, True
        else:
            # Incomplete cross-validation run, resume cross-validation
            print(f"Found incomplete run folder: {run_folder_path}. Resuming cross-validation in this folder.")
            return run_folder_path, True

    # If no incomplete runs found, create a new run folder
    next_run = max(run_numbers) + 1 if run_numbers else 1
    run_folder_name = f"{base_name}_{next_run}"
    run_folder_path = os.path.join(outputs_dir, run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)
    print(f"Starting a new training run. Saving outputs to: {run_folder_path}")
    return run_folder_path, False  # New run


if __name__ == "__main__":
    # Determine the next run folder and whether it's a resumed run
    run_folder, is_resumed = get_next_run_folder(outputs_dir='./outputs', base_name='rdecnn_with_resnet')
    
    # Initialize classifier with the run folder as save_path
    classifier = FashionMNIST_RDECNNClassifier(batch_size=64, lr=1e-3, epochs=40, save_path=run_folder)
    
    # Log and print whether training is being resumed or started fresh
    if is_resumed:
        classifier.logger.info(f"=== Resuming training in run folder: {run_folder} ===")
        print(f"=== Resuming training in run folder: {run_folder} ===")
    else:
        classifier.logger.info(f"=== Starting new training run in folder: {run_folder} ===")
        print(f"=== Starting new training run in folder: {run_folder} ===")

    print("\n=== Model Structure ===")
    print(classifier.model)  # Print the model structure

    # --- Perform Cross-Validation and Standard Training ---
    classifier.train_with_cross_validation_and_standard(k_folds=5)

    # --- Calculate and Print Cross-Validation Statistics ---
    classifier.calculate_all_cv_statistics()

    # --- Save All Metrics after Cross-Validation and Testing ---
    classifier.save_all_metrics()
    print("All training scenarios have been processed. Metrics saved.")
