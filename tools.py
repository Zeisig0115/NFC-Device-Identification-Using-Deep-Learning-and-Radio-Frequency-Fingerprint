import h5py
import os
import re
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math


class MyDataset(Dataset):
    """
    Custom dataset for loading signals from HDF5 files.

    Each HDF5 file is expected to contain a dataset named 'signals'.
    The file name should include the tag label as 'tagX', where X is the label (1-indexed).
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Root directory containing HDF5 files.
        """
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []

        # Loop through all files in the root directory
        for filename in os.listdir(root_dir):
            if filename.endswith('.hdf5'):
                # Extract label from the filename (convert to 0-indexed)
                label = int(re.search(r'tag(\d+)', filename).group(1)) - 1
                file_path = os.path.join(root_dir, filename)
                with h5py.File(file_path, 'r') as hdf:
                    num_signals = len(hdf['signals'])
                    self.file_paths.extend([file_path] * num_signals)
                    self.labels.extend([label] * num_signals)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        with h5py.File(file_path, 'r') as hdf:
            # Use modulo in case the index exceeds the number of signals in the file
            data = hdf['signals'][idx % len(hdf['signals'])]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor


class EarlyStopping:
    """
    Early stopping to stop training when the validation loss stops improving.
    """

    def __init__(self, patience=10, verbose=False, path='./checkpoint.pth'):
        """
        Args:
            patience (int): How many epochs to wait before stopping when no improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path to save the checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.val_loss_min = float('inf')
        self.path = path
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        """
        Check if validation loss has decreased; if not, increment counter.
        If counter exceeds patience, trigger early stopping.
        """
        if val_loss < self.val_loss_min:
            self.save_checkpoint(val_loss, model, epoch)
            self.val_loss_min = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stop training — The loss has not improved after {self.patience} epochs.")

    def save_checkpoint(self, val_loss, model, epoch):
        """
        Save model when validation loss decreases.
        """
        if self.verbose:
            print(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training.

    Returns an anchor, a positive example (same class), and a negative example (different class).
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Directory containing HDF5 files with signal data.
        """
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.label_to_indices = {}

        for filename in os.listdir(root_dir):
            if filename.endswith('.hdf5'):
                # Extract label from filename (0-indexed)
                label = int(re.search(r'tag(\d+)', filename).group(1)) - 1
                file_path = os.path.join(root_dir, filename)
                with h5py.File(file_path, 'r') as hdf:
                    signals = hdf['signals'][:]
                    for signal in signals:
                        idx = len(self.data)
                        self.data.append(signal)
                        self.labels.append(label)
                        if label not in self.label_to_indices:
                            self.label_to_indices[label] = []
                        self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_anchor = self.data[idx]
        label = self.labels[idx]
        # Choose a positive example (different from anchor)
        positive_index = idx
        while positive_index == idx:
            positive_index = np.random.choice(self.label_to_indices[label])
        # Choose a negative example (from a different class)
        negative_label = np.random.choice(list(set(self.labels) - {label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        data_positive = self.data[positive_index]
        data_negative = self.data[negative_index]

        # Convert data to tensors
        data_anchor = torch.tensor(data_anchor, dtype=torch.float32)
        data_positive = torch.tensor(data_positive, dtype=torch.float32)
        data_negative = torch.tensor(data_negative, dtype=torch.float32)

        return data_anchor, data_positive, data_negative, label


class TripletLoss(nn.Module):
    """
    Triplet loss function.

    Uses the hardest positive and negative examples in the batch to compute the margin ranking loss.
    """

    def __init__(self, margin=0.2):
        """
        Args:
            margin (float): Margin for the triplet loss.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Embeddings with shape (batch_size, embedding_dim).
            targets (torch.Tensor): Ground truth labels for the embeddings.

        Returns:
            torch.Tensor: The computed triplet loss.
        """
        n = inputs.size(0)  # Batch size

        # Compute pairwise Euclidean distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hardest positive: maximum distance among same-class samples
            hard_positive = dist[i][mask[i]].max().unsqueeze(0)
            # Hardest negative: minimum distance among different-class samples
            hard_negative = dist[i][mask[i] == 0].min().unsqueeze(0)
            dist_ap.append(hard_positive)
            dist_an.append(hard_negative)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class SoftTriple(nn.Module):
    """
    SoftTriple loss function.

    This loss function aims to improve the discriminative power of deep features
    by learning multiple centers for each class.
    """

    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        """
        Args:
            la (float): Scaling factor.
            gamma (float): Inverse temperature for softmax.
            tau (float): Regularization weight.
            margin (float): Margin parameter.
            dim (int): Dimension of the embeddings.
            cN (int): Number of classes.
            K (int): Number of centers per class.
        """
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1.0 / gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        # Parameter: centers for each class, shape: (dim, cN*K)
        self.fc = Parameter(torch.Tensor(dim, cN * K))
        # Create a binary mask for center regularization
        self.weight = torch.zeros(cN * K, cN * K, dtype=torch.bool).cuda()
        for i in range(cN):
            for j in range(K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        self.to(torch.device('cuda'))

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Input embeddings of shape (batch_size, dim).
            target (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The computed SoftTriple loss.
        """
        # Normalize centers
        centers = F.normalize(self.fc, p=2, dim=0)
        # Compute similarity between input embeddings and centers
        simInd = input.matmul(centers)
        # Reshape similarities to (batch_size, cN, K)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        # Compute softmax probabilities along centers per class
        prob = F.softmax(simStruc * self.gamma, dim=2)
        # Aggregate similarity for each class
        simClass = torch.sum(prob * simStruc, dim=2)
        # Create margin mask
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin

        lossClassify = F.cross_entropy(self.la * (simClass - marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2.0 * simCenter[self.weight])) / (self.cN * self.K * (self.K - 1.))
            return lossClassify + self.tau * reg
        else:
            return lossClassify
