from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

from models import model_1_tri, model_2_tri
from tools import MyDataset, EarlyStopping, TripletLoss, SoftTriple
from demo1 import testset_eval, get_embeddings


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


# Set seed for reproducibility
setup_seed(1)

# Hyper Parameters
num_epochs = 200
LR = 1e-4
patience = 20
batch_size = 256
train_ratio = 0.9
margin = 0.2
decayRate = 0.987
model_save_path = './trained_net/model_1_ST.pth'

# Initialize model and optimizer
model = model_1_tri().cuda()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=0.0001)

# Loss function: Uncomment the desired loss function.
# loss_function = TripletLoss(margin=margin)
loss_function = SoftTriple(la=0.5, gamma=0.1, tau=0.2, margin=margin, dim=512, cN=8, K=1)

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# Prepare dataset and split into train/validation sets
dataset = MyDataset(root_dir='./data/d1234')
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def adjust_learning_rate(optimizer, epoch, warmup_epochs=20, base_lr=LR):
    """
    Adjusts the learning rate based on the current epoch.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be adjusted.
        epoch (int): Current epoch.
        warmup_epochs (int): Number of warm-up epochs.
        base_lr (float): Base learning rate.
    """
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr_scheduler.step()


def train(model, train_loader, val_loader, optimizer, triplet_loss, num_epochs=50, model_save_path=None):
    """
    Train the model using the given training and validation loaders.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        triplet_loss (callable): Loss function.
        num_epochs (int): Number of epochs.
        model_save_path (str): Path to save the best model.
    """
    with open('results.txt', 'w') as file:
        for epoch in range(num_epochs):
            adjust_learning_rate(optimizer, epoch, warmup_epochs=20, base_lr=LR)
            model.train()
            total_train_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.unsqueeze(1).cuda()  # Add channel dimension if necessary
                labels = labels.cuda()
                optimizer.zero_grad()
                embed = model(inputs)
                loss = triplet_loss(embed, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.unsqueeze(1).cuda()
                    labels = labels.cuda()
                    embed = model(inputs)
                    loss = triplet_loss(embed, labels)
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Train Total Loss: {avg_train_loss:.4f}')
            print(f'Validation Total Loss: {avg_val_loss:.4f}')
            print('==' * 30)
            file.write(f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

            early_stopping(avg_val_loss, model, epoch + 1)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {early_stopping.best_epoch}.")
                print(
                    f"Loading best model from epoch {early_stopping.best_epoch} with loss {early_stopping.val_loss_min}")
                break


def test():
    """
    Load the best model and evaluate using a k-NN classifier.
    """
    model = model_1_tri()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    # Get embeddings for the training set
    Train_embeddings, Train_labels = get_embeddings(model, train_loader)
    k = 1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Train_embeddings, Train_labels)
    testset_eval(Tri_model=model, test_loader=val_loader, knn=knn)


# Uncomment the following line to train the model:
# train(model, train_loader, val_loader, optimizer, loss_function, num_epochs, model_save_path)

# Test the model
test()
