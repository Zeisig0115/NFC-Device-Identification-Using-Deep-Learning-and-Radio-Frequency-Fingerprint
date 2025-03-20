import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tools import MyDataset
from models import model_2_feature, model_2_tri

# Load test and train datasets
test_dataset = MyDataset(root_dir='./data/d5678')
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

train_dataset = MyDataset(root_dir='./data/d1234')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# Define model paths
CE_model_path = './trained_net/model_2_ce.pth'
Tri_model_path = './trained_net/model_2_ST.pth'

num_classes = 8

# Initialize models and load pretrained weights
CE_model = model_2_feature(num_classes).cuda()
Tri_model = model_2_tri().cuda()

CE_model.load_state_dict(torch.load(CE_model_path))
Tri_model.load_state_dict(torch.load(Tri_model_path))

CE_model.eval()
Tri_model.eval()


def extract_features_and_labels(model, loader):
    """
    Extract feature embeddings and corresponding labels using the given model.

    Args:
        model (torch.nn.Module): Feature extraction model.
        loader (DataLoader): DataLoader for the dataset.

    Returns:
        tuple: (features, labels) as numpy arrays.
    """
    features = []
    labels = []

    with torch.no_grad():
        for data, target in loader:
            data = data.unsqueeze(1).cuda()  # Add channel dimension
            output = model(data)
            features.extend(output.cpu().numpy())
            labels.extend(target.numpy())

    return np.array(features), np.array(labels)


# Extract features from train and test sets using both models
CE_train_features, CE_train_labels = extract_features_and_labels(CE_model, train_loader)
Tri_train_features, Tri_train_labels = extract_features_and_labels(Tri_model, train_loader)
CE_test_features, CE_test_labels = extract_features_and_labels(CE_model, test_loader)
Tri_test_features, Tri_test_labels = extract_features_and_labels(Tri_model, test_loader)

# Remove any extra dimensions if needed
CE_train_features = CE_train_features.squeeze()
Tri_train_features = Tri_train_features.squeeze()


def visualize_features_subplot(CE_features, Tri_features, labels):
    """
    Visualize features using t-SNE in two subplots.

    Args:
        CE_features (np.array): Features from the CE model.
        Tri_features (np.array): Features from the triplet loss model.
        labels (np.array): Ground truth labels.
    """
    tsne = TSNE(n_components=2, random_state=0)
    CE_tsne_results = tsne.fit_transform(CE_features)
    Tri_tsne_results = tsne.fit_transform(Tri_features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i in range(num_classes):
        axes[0].scatter(CE_tsne_results[labels == i, 0],
                        CE_tsne_results[labels == i, 1],
                        label=f"Tag {i + 1}")
        axes[1].scatter(Tri_tsne_results[labels == i, 0],
                        Tri_tsne_results[labels == i, 1],
                        label=f"Tag {i + 1}")

    axes[0].set_title("CE Model Features", fontsize=20)
    axes[1].set_title("Triplet Model Features", fontsize=20)
    axes[0].legend()
    axes[1].legend()
    plt.show()


# Uncomment the following line to visualize the features with t-SNE:
# visualize_features_subplot(CE_train_features, Tri_train_features, Tri_train_labels)


def calculate_recall_corrected(features, labels, num_classes):
    """
    Calculate a corrected recall value per class based on Euclidean distances.

    For each sample, the nearest neighbor (excluding itself) is retrieved.
    If the nearest neighbor has the same label, it is considered a correct retrieval.

    Args:
        features (np.array): Feature embeddings.
        labels (np.array): Ground truth labels.
        num_classes (int): Total number of classes.

    Returns:
        np.array: Recall values per class.
    """
    # Compute pairwise Euclidean distances and set self-distances to infinity
    dists = cdist(features, features, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    recall_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        if len(class_indices) > 1:
            correct_retrievals = 0
            for idx in class_indices:
                nearest_idx = np.argmin(dists[idx, :])
                if labels[nearest_idx] == i:
                    correct_retrievals += 1
            recall_per_class[i] = correct_retrievals / len(class_indices)
    return recall_per_class


# Compute corrected recall values for both models
recall_ce_corrected = 100 * calculate_recall_corrected(CE_test_features, CE_test_labels, num_classes)
recall_tri_corrected = 100 * calculate_recall_corrected(Tri_test_features, Tri_test_labels, num_classes)

# Print recall values per class
for i in range(num_classes):
    print(f"Class {i}: Corrected CE Recall = {recall_ce_corrected[i]:.3f}, "
          f"Corrected Triplet Recall = {recall_tri_corrected[i]:.3f}")


def plot_recall_comparison(recall_ce, recall_tri, num_classes):
    """
    Plot a bar chart comparing recall for CE and Triplet models by class.

    Args:
        recall_ce (np.array): Recall values for the CE model.
        recall_tri (np.array): Recall values for the Triplet model.
        num_classes (int): Number of classes.
    """
    labels = range(num_classes)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, recall_ce, width, label='CE Model')
    rects2 = ax.bar(x + width/2, recall_tri, width, label='Triplet Model')

    ax.set_ylabel('Recall')
    ax.set_title('Recall by Class and Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{round(height,2)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()


# Plot the recall comparison for both models
plot_recall_comparison(recall_ce_corrected, recall_tri_corrected, num_classes)
