import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
import json
import scipy.io.wavfile as wav
from scipy.spatial.distance import pdist, cdist
import time

from tools import MyDataset
from models import model_1_tri, model_2_tri
from methods import normalize_method3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def testset_eval(Tri_model, test_loader, knn):
    """
    Evaluate the Tri_model on the test set using a k-NN classifier.

    Args:
        Tri_model (torch.nn.Module): The model used to generate embeddings.
        test_loader (DataLoader): DataLoader for the test dataset.
        knn (KNeighborsClassifier): k-NN classifier trained on registered embeddings.

    Prints:
        Test set accuracy and displays the confusion matrix.
    """
    test_embeddings, test_labels = get_embeddings(Tri_model, test_loader)
    test_labels_predicted = knn.predict(test_embeddings)
    test_accuracy = accuracy_score(test_labels, test_labels_predicted)
    print(f"Test set accuracy: {test_accuracy}")
    cm = confusion_matrix(test_labels, test_labels_predicted)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.title('Confusion Matrix', fontsize=20)
    plt.show()


def compute_thresholds(registered_embeddings, registered_labels):
    """
    Compute distance thresholds for each class based on the registered embeddings.

    For each unique label, calculate the pairwise Euclidean distances, then set the
    threshold as the median distance plus 1.5 times the standard deviation.

    Args:
        registered_embeddings (np.array): Array of embeddings.
        registered_labels (np.array): Array of corresponding labels.

    Returns:
        dict: A dictionary mapping each label to its computed threshold.
    """
    thresholds = {}
    for label in np.unique(registered_labels):
        label_embeddings = registered_embeddings[registered_labels == label]
        distances = pdist(label_embeddings, 'euclidean')
        median = np.median(distances)
        std = np.std(distances)
        thresholds[label] = median + 1.5 * std
    return thresholds


def predict():
    """
    Predict whether each sample in the APP folder is genuine or anomalous.

    Loads a signal file and corresponding JSON file, extracts signal segments,
    normalizes them, obtains embeddings, and uses a k-NN classifier to predict the
    label. Then, compares the mean distance of the embedding to a pre-computed
    threshold to decide if the sample is anomalous.
    """
    thresholds = compute_thresholds(Train_embeddings, Train_labels)
    folder_path = './APP'
    signal_dir = None
    json_dir = None

    # Find the first .wav and .json file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') and signal_dir is None:
            signal_dir = os.path.join(folder_path, filename)
        elif filename.endswith('.json') and json_dir is None:
            json_dir = os.path.join(folder_path, filename)
        if signal_dir is not None and json_dir is not None:
            break

    # Read the signal from the wav file
    _, signal = wav.read(signal_dir)

    # Extract atqa sample start positions from the JSON file
    atqa_starts = []
    with open(json_dir, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        frames = json_data['frames']
        for frame in frames:
            if "04:00" in frame['frameData']:
                atqa_starts.append(frame['sampleStart'])

    genuine_count = 0
    anomaly_count = 0
    predictions = []

    # Iterate over signal segments
    for i in range(len(atqa_starts) - 1):
        atqa_signal = signal[atqa_starts[i]:atqa_starts[i] + 410]
        norm_signal = normalize_method3(atqa_signal)
        # Prepare input tensor with necessary dimensions
        inp = torch.tensor(norm_signal).unsqueeze(0).unsqueeze(1).float()
        test_embedding = Tri_model(inp)
        test_embedding = test_embedding.detach().cpu()
        predicted_label = knn.predict(test_embedding)[0]
        predictions.append(predicted_label)
        # Retrieve indices for the predicted label in the k-NN classes
        label_indices = np.where(knn.classes_ == predicted_label)[0]
        distances, _ = knn.kneighbors(test_embedding, n_neighbors=len(label_indices))
        distances = distances.flatten()
        mean_distance = np.mean(distances)
        if mean_distance > thresholds[predicted_label]:
            anomaly_count += 1
        else:
            genuine_count += 1

    # Bar plot for genuine vs anomaly counts
    categories = ['Genuine', 'Anomaly']
    quantities = [genuine_count, anomaly_count]
    plt.figure(figsize=(8, 6))
    bar_positions = np.arange(len(categories))
    plt.bar(bar_positions, quantities, color=['skyblue', 'salmon'])
    plt.xticks(bar_positions, categories)
    plt.ylabel('Count')
    plt.title('Prediction of Genuine & Anomaly Packets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Uncomment the following line to display the bar plot:
    # plt.show()

    # If genuine count is higher than anomaly, show histogram of predicted classes
    if anomaly_count < genuine_count:
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(
            predictions, bins=np.arange(-0.5, 8.5, 1),
            edgecolor='black', rwidth=0.8
        )
        for count, patch in zip(counts, patches):
            height = patch.get_height()
            plt.text(patch.get_x() + patch.get_width() / 2, height + 0.1,
                     f'{int(count)}', ha='center', va='bottom', fontsize=12)
        plt.xticks(range(8), ['Tag 1', 'Tag 2', 'Tag 3', 'Tag 4',
                              'Tag 5', 'Tag 6', 'Tag 7', 'Tag 8'])
        plt.title('Total Sample Predicted: {}'.format(len(predictions)))
        plt.xlabel('Predicted Class')
        plt.ylabel('Number of Predictions')
        # Uncomment the following line to display the histogram:
        # plt.show()


def get_embeddings(model, dataloader):
    """
    Compute embeddings for all samples in the dataloader using the provided model.

    Args:
        model (torch.nn.Module): Model used to extract embeddings.
        dataloader (DataLoader): DataLoader for the dataset.

    Returns:
        tuple: A tuple (embeddings, labels) where embeddings is a numpy array of shape
               (num_samples, embedding_dim) and labels is a numpy array of corresponding labels.
    """
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, classes in dataloader:
            inputs = inputs.unsqueeze(1)
            emb = model(inputs)
            embeddings.append(emb.numpy())
            labels.extend(classes.numpy())
    return np.vstack(embeddings), np.array(labels)


# Load training and testing datasets
train_dataset = MyDataset(root_dir='./data/d1234')
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
test_dataset = MyDataset(root_dir='./data/d5678')
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Uncomment the following block to use a random split of a single dataset:
# train_ratio = 0.01
# dataset = MyDataset(root_dir='./data/d1234')
# train_size = int(train_ratio * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


if __name__ == '__main__':
    # Load Tri_model and its weights
    Tri_model_path = './trained_net/model_2_TRI.pth'
    Tri_model = model_2_tri()
    Tri_model.load_state_dict(torch.load(Tri_model_path))
    Tri_model.eval()

    # Compute embeddings for the training set
    Train_embeddings, Train_labels = get_embeddings(Tri_model, train_loader)

    # Train a k-NN classifier on the training embeddings
    k = 1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Train_embeddings, Train_labels)

    # Optionally evaluate the model on a test set
    # testset_eval(Tri_model, test_loader, knn)

    # Run prediction on data from the APP folder
    predict()
