import torch
import time
import os
import logging
import random
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from CNN import CNN  # Ensure this is correctly imported from your project
from Polluted_Images_Generation import CRRNWEP  # Ensure this is correctly imported

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f'Validation loss decreased to {val_loss:.4f}. Saving model.')

class FashionMNIST_CNNClassifier:
    def __init__(self, batch_size=64, lr=1e-3, epochs=100, patience=10, device=None, save_path='../models/CNN'):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience  # For early stopping
        if device:
            self.device = device
        else:
            # Safely check for MPS support (for macOS with Apple Silicon)
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()  # Clears the MPS cache
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")        
        self.save_path = save_path

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

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()

        print('Using device:', self.device)

    def set_path(self, path):
        self.save_path = path

    def load_and_split_data(self):
        """
        Loads the FashionMNIST dataset and splits it into training, validation, and test sets (70-15-15).
        This is done for normal, polluted, and mixed datasets.
        """
        # Load datasets
        train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=self.transform)

        polluted_train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=self.polluting_transform)
        polluted_test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=self.polluting_transform)

        # Mixed dataset: concatenation of normal and polluted training data
        mixed_train_dataset = ConcatDataset([train_dataset, polluted_train_dataset])

        # Define train-validation-test splits
        def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
            total_size = len(dataset)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            return random_split(dataset, [train_size, val_size, test_size])

        # Split normal training dataset
        normal_train, normal_val, normal_test_split = split_dataset(train_dataset)
        # Split polluted training dataset
        polluted_train, polluted_val, polluted_test_split = split_dataset(polluted_train_dataset)
        # Split mixed training dataset
        mixed_train, mixed_val, mixed_test_split = split_dataset(mixed_train_dataset)

        # Define DataLoaders
        self.train_loaders = {
            'normal': DataLoader(normal_train, batch_size=self.batch_size, shuffle=True),
            'polluted': DataLoader(polluted_train, batch_size=self.batch_size, shuffle=True),
            'mixed': DataLoader(mixed_train, batch_size=self.batch_size, shuffle=True)
        }

        self.val_loaders = {
            'normal': DataLoader(normal_val, batch_size=self.batch_size, shuffle=False),
            'polluted': DataLoader(polluted_val, batch_size=self.batch_size, shuffle=False),
            'mixed': DataLoader(mixed_val, batch_size=self.batch_size, shuffle=False)
        }

        self.test_loaders = {
            'normal': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False),
            'polluted': DataLoader(polluted_test_dataset, batch_size=self.batch_size, shuffle=False),
            'mixed': DataLoader(mixed_test_split, batch_size=self.batch_size, shuffle=False)  # Assuming mixed test set
        }

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
        )
        model = CNN(**self.params).to(self.device)
        return model

    def train_one_epoch(self, model, optimizer, loader, val_loader, epoch, scenario, early_stopping):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        correct, total = 0, 0

        with tqdm(loader, desc=f'{scenario.capitalize()} Training Epoch {epoch+1}/{self.epochs}') as pbar:
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Compute loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100 * correct / total
        epoch_duration = time.time() - start_time

        # Validation
        val_loss, val_accuracy = self.validate(model, val_loader, scenario)

        # Logging
        logging.info(f"Scenario: {scenario}, Epoch {epoch+1}/{self.epochs}, "
                     f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                     f"Duration: {epoch_duration:.2f}s")
        print(f"Scenario: {scenario}, Epoch {epoch+1}/{self.epochs}, "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
              f"Duration: {epoch_duration:.2f}s")

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered for scenario: {scenario} at epoch {epoch+1}")
            print(f"Early stopping triggered for scenario: {scenario} at epoch {epoch+1}")
            return True  # Indicate that training should stop

        return False  # Continue training

    def validate(self, model, val_loader, scenario):
        model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'{scenario.capitalize()} Validation', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def test_model(self, model, loader, scenario):
        model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f'{scenario.capitalize()} Testing', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100 * correct / total
        logging.info(f"Scenario: {scenario}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        print(f"Scenario: {scenario}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

        # Save test results
        result_path = os.path.join('cnn_results', f"{scenario}_test_results.txt")
        with open(result_path, 'w') as f:
            f.write(f"Test Loss: {avg_loss:.4f}\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n")

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        print("Loaded model from", path)

def setup_logging(results_dir):
    log_file = os.path.join(results_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    # Create cnn_results directory
    results_dir = 'cnn_results'
    os.makedirs(results_dir, exist_ok=True)

    # Setup logging
    setup_logging(results_dir)
    logging.info("Starting training and testing scenarios.")

    # Initialize classifier
    classifier = FashionMNIST_CNNClassifier(batch_size=64, lr=1e-3, epochs=50, patience=10)
    
    # Note: The model is not stored as classifier.model in this refactored version
    # Remove or comment out the following line to avoid AttributeError
    # print(classifier.model)

    # Define training and corresponding validation loaders
    train_scenarios = {
        'normal': (classifier.train_loaders['normal'], classifier.val_loaders['normal']),
        'polluted': (classifier.train_loaders['polluted'], classifier.val_loaders['polluted']),
        'mixed': (classifier.train_loaders['mixed'], classifier.val_loaders['mixed'])
    }

    # Define test scenarios
    test_scenarios = {
        'normal': classifier.test_loaders['normal'],
        'polluted': classifier.test_loaders['polluted']
        # If you have a distinct 'mixed' test set, add it here
        # 'mixed': classifier.test_loaders['mixed']
    }

    for train_scenario, (train_loader, val_loader) in train_scenarios.items():
        logging.info(f"Starting training with {train_scenario} data.")
        print(f"Starting training with {train_scenario} data.")

        # Initialize a fresh model and optimizer for each training scenario
        model = classifier.initialize_model()
        optimizer = optim.Adam(model.parameters(), lr=classifier.lr)

        # Initialize Early Stopping
        checkpoint_path = os.path.join(results_dir, f"{train_scenario}_best_model.pt")
        early_stopping = EarlyStopping(patience=classifier.patience, verbose=True, save_path=checkpoint_path)

        # Training Loop
        for epoch in range(classifier.epochs):
            stop = classifier.train_one_epoch(model, optimizer, train_loader, val_loader, epoch, train_scenario, early_stopping)
            if stop:
                break  # Early stopping triggered

        # Load the best model
        classifier.load_model(model, checkpoint_path)

        # Define which test scenarios to evaluate after each training
        applicable_tests = ['normal', 'polluted']  # Adjust if needed

        for test_scenario in applicable_tests:
            logging.info(f"Testing model trained on {train_scenario} data on {test_scenario} test data.")
            print(f"Testing model trained on {train_scenario} data on {test_scenario} test data.")
            classifier.test_model(model, test_scenarios[test_scenario], f"{train_scenario}_{test_scenario}")

    logging.info("Completed all training and testing scenarios.")
    print("Completed all training and testing scenarios.")

if __name__ == "__main__":
    main()
