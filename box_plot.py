import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc

from tools import MyDataset
from models import model_1_tri, model_2_tri, model_2_feature, model_1_feature


def get_embeddings(model, dataloader):
    """
    Compute embeddings for all samples in a dataloader using the provided model.

    Args:
        model (torch.nn.Module): The model to extract embeddings.
        dataloader (DataLoader): DataLoader for the dataset.

    Returns:
        tuple: A tuple (embeddings, labels) where embeddings is a numpy array
               of shape (num_samples, embedding_dim) and labels is a numpy array
               of the corresponding labels.
    """
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, classes in dataloader:
            # Add channel dimension and move to GPU
            inputs = inputs.unsqueeze(1).cuda()
            emb = model(inputs)
            embeddings.append(emb.cpu().numpy())
            labels.extend(classes.numpy())
    return np.vstack(embeddings), np.array(labels)


# Prepare test dataset and dataloader
test_dataset = MyDataset(root_dir='./data/d1234')
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define model paths
CE_model_path = './trained_net/model_2_CE.pth'
Tri_model_path = './trained_net/model_2_ST.pth'

num_classes = 8

# Initialize models and move them to GPU
CE_model = model_2_feature(num_classes).cuda()
Tri_model = model_2_tri().cuda()

# Load pre-trained weights
CE_model.load_state_dict(torch.load(CE_model_path))
Tri_model.load_state_dict(torch.load(Tri_model_path))
CE_model.eval()
Tri_model.eval()

# Get embeddings and labels from both models
CE_embeddings, CE_labels = get_embeddings(CE_model, test_loader)
Tri_embeddings, Tri_labels = get_embeddings(Tri_model, test_loader)


def calculate_distance(embeddings, labels, num_samples=40):
    """
    Calculate genuine and forged distances for each class.

    For each class (genuine tag), a set of embeddings is randomly selected.
    For each selected embedding, the Euclidean distance is computed against all
    embeddings of the same class (genuine distances) and against a random subset
    of embeddings from all other classes (forged distances).

    Args:
        embeddings (np.array): Embeddings of shape (num_samples, embedding_dim).
        labels (np.array): Array of labels corresponding to each embedding.
        num_samples (int): Number of samples to randomly select per class.

    Returns:
        tuple: Two lists containing genuine distances and forged distances for each class.
    """
    all_genuine_distances = []
    all_forged_distances = []

    # Iterate over each class
    for genuine_class_id in range(num_classes):
        # Get embeddings for the genuine class
        genuine_class_embeddings = embeddings[labels == genuine_class_id]

        # Randomly select num_samples embeddings for genuine class if possible
        if len(genuine_class_embeddings) >= num_samples:
            indices = random.sample(range(len(genuine_class_embeddings)), num_samples)
            selected_genuine_embeddings = genuine_class_embeddings[indices]
        else:
            selected_genuine_embeddings = genuine_class_embeddings

        genuine_distances = []
        forged_distances = []

        # Calculate distances for each selected genuine embedding
        for selected_emb in selected_genuine_embeddings:
            # Genuine distances within the same class
            distances = np.linalg.norm(selected_emb - genuine_class_embeddings, axis=1)
            genuine_distances.extend(distances)

            # Forged distances: distances to embeddings from all other classes
            for forged_class_id in range(num_classes):
                if forged_class_id != genuine_class_id:
                    forged_class_embeddings = embeddings[labels == forged_class_id]
                    if len(forged_class_embeddings) >= num_samples:
                        indices = random.sample(range(len(forged_class_embeddings)), num_samples)
                        selected_forged_embeddings = forged_class_embeddings[indices]
                    else:
                        selected_forged_embeddings = forged_class_embeddings
                    distances = np.linalg.norm(selected_emb - selected_forged_embeddings, axis=1)
                    forged_distances.extend(distances)

        all_genuine_distances.append(genuine_distances)
        all_forged_distances.append(forged_distances)

    return all_genuine_distances, all_forged_distances


def box_plot(all_genuine_distances, all_forged_distances):
    """
    Create a box plot to visualize genuine and forged distances for each class.

    Genuine distances are plotted in red and forged distances in blue.

    Args:
        all_genuine_distances (list): List of genuine distances for each class.
        all_forged_distances (list): List of forged distances for each class.
    """
    fig, ax = plt.subplots()
    positions = np.array(range(num_classes)) * 2
    genuine_positions = positions - 0.4
    forged_positions = positions + 0.4

    # Plot genuine distances (same tag) in red
    plt.boxplot(
        all_genuine_distances,
        positions=genuine_positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="red", color="red"),
        medianprops=dict(color="yellow"),
        showfliers=True,
        flierprops=dict(markerfacecolor='g', marker='o')
    )

    # Plot forged distances (different tags) in blue
    plt.boxplot(
        all_forged_distances,
        positions=forged_positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="blue", color="blue"),
        medianprops=dict(color="yellow"),
        showfliers=True,
        flierprops=dict(markerfacecolor='g', marker='o')
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f'Tag {i + 1}' for i in range(num_classes)])
    ax.set_ylabel('Distances', fontsize=20)
    ax.set_title('Distances by Device', fontsize=20)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Same Tag'),
        Patch(facecolor='blue', label='Different Tags')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()


# Calculate distances using the Tri_model embeddings with num_samples=400 per class
all_genuine_distances, all_forged_distances = calculate_distance(Tri_embeddings, Tri_labels, num_samples=400)

# Display the box plot
box_plot(all_genuine_distances, all_forged_distances)
