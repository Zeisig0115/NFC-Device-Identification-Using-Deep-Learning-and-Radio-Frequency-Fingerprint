import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tools import MyDataset, EarlyStopping
from models import model_1, model_2


def setup_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set the random seed
setup_seed(1)

# Prepare dataset and split into training and validation sets
dataset = MyDataset(root_dir='./data/d1234')
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Set device to GPU
device = torch.device("cuda")

# Define model parameters and initialize the model
num_classes = 8
model = model_2(num_classes).to(device)
model_save_path = './trained_net/model_2_CE.pth'

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Alternatively, use SGD:
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.0001)

# Define learning rate scheduler and early stopping
decayRate = 0.987
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# Alternatively, you can use:
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=32)
early_stopping = EarlyStopping(patience=20, verbose=True, path=model_save_path)

num_epochs = 200


def adjust_learning_rate(optimizer, epoch, warmup_epochs=20, base_lr=1e-4):
    """
    Adjusts the learning rate for each epoch based on warmup strategy.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate to adjust.
        epoch (int): Current epoch.
        warmup_epochs (int): Number of epochs to use for warmup.
        base_lr (float): Base learning rate.
    """
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train():
    """
    Train the model on the training set and evaluate on the validation set.

    Logs losses and accuracies per epoch and applies early stopping.
    """
    with open('results.txt', 'w') as file:
        for epoch in range(num_epochs):
            train_losses = []
            test_losses = []
            train_accs = []
            test_accs = []

            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)  # Add channel dimension if needed
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = correct_train / total_train
            average_train_loss = train_loss / len(train_loader)

            # Validation loop
            correct_test = 0
            total_test = 0
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            test_accuracy = correct_test / total_test
            average_test_loss = test_loss / len(val_loader)

            # Log results to file and console
            file.write(
                f'{epoch + 1} {average_train_loss:.4f} {train_accuracy * 100:.2f} '
                f'{average_test_loss:.4f} {test_accuracy * 100:.2f}\n'
            )
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%')
            print(f'Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

            train_losses.append(average_train_loss)
            test_losses.append(average_test_loss)
            train_accs.append(train_accuracy)
            test_accs.append(test_accuracy)

            # Step the learning rate scheduler
            lr_scheduler.step()

            # Check early stopping condition
            early_stopping(average_test_loss, model, epoch + 1)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {early_stopping.best_epoch}.")
                print(
                    f"Loading best model from epoch {early_stopping.best_epoch} with loss {early_stopping.val_loss_min}"
                )
                break


def test():
    """
    Test the trained model on the validation set.

    Loads the best model saved during training, computes predictions on the
    validation set, calculates accuracy, and plots a confusion matrix.
    """
    model.load_state_dict(torch.load(model_save_path))
    correct = 0
    total = 0
    model.eval()
    all_predicted = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.tolist())
            all_true_labels.extend(labels.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the new data: {accuracy}%')

    confusion = confusion_matrix(all_true_labels, all_predicted)
    plt.figure(figsize=(num_classes, num_classes))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    train()
    test()
