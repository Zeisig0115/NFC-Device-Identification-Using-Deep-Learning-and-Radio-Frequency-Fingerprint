import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class model_1(nn.Module):
    """
    A convolutional model for classification.

    Architecture:
        - Three 1D convolutional layers with BatchNorm and ReLU.
        - Adaptive average pooling.
        - Fully connected layer to 512 dimensions.
        - L2 normalization.
        - Dropout.
        - Final classification layer.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes.
        """
        super(model_1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: Logits for classification.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class model_1_feature(nn.Module):
    """
    A feature extraction variant of model_1.

    Architecture is similar to model_1, but without the final dropout and classification layer.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes (not used in forward).
        """
        super(model_1_feature, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass to obtain features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: L2-normalized feature embedding.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class model_1_tri(nn.Module):
    """
    A triplet-based feature extraction model.

    Similar architecture as model_1_feature but without dropout and final classification layer.
    """

    def __init__(self):
        super(model_1_tri, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 512)

    def forward(self, x):
        """
        Forward pass to obtain triplet embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: L2-normalized embedding.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ResBlock_model2(nn.Module):
    """
    A residual block for model_2.

    Consists of two 1D convolutional layers with BatchNorm and ReLU,
    plus a residual connection.
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ResBlock_model2, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        # If dimensions differ, use a 1x1 convolution to match dimensions.
        if in_channels != out_channels:
            self.res_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                       stride=1, padding=0)
        else:
            self.res_layer = None

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual addition and ReLU.
        """
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return F.relu(self.layer(x) + residual)


class model_2(nn.Module):
    """
    A deeper residual network for classification.

    Architecture:
        - Initial convolution layer.
        - Four residual blocks.
        - Adaptive average pooling.
        - Fully connected dense layer to 512 dimensions.
        - L2 normalization.
        - Dropout.
        - Final classification layer.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes.
        """
        super(model_2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=7, stride=2, padding=3)
        self.block1 = ResBlock_model2(32, 32)
        self.block2 = ResBlock_model2(32, 32)

        self.block3 = ResBlock_model2(32, 64)
        self.block4 = ResBlock_model2(64, 64)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(64, 512)

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class model_2_feature(nn.Module):
    """
    A feature extraction variant of model_2.

    Architecture similar to model_2, but without the dropout and final classification layer.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes (not used in forward).
        """
        super(model_2_feature, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=7, stride=2, padding=3)
        self.block1 = ResBlock_model2(32, 32)
        self.block2 = ResBlock_model2(32, 32)
        self.block3 = ResBlock_model2(32, 64)
        self.block4 = ResBlock_model2(64, 64)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(64, 512)

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass to obtain features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: L2-normalized feature embedding.
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x


class model_2_tri(nn.Module):
    """
    A triplet-based feature extraction model based on model_2.

    Similar to model_2_feature, but without dropout and final classification layers.
    """

    def __init__(self):
        super(model_2_tri, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,
                               kernel_size=7, stride=2, padding=3)
        self.block1 = ResBlock_model2(32, 32)
        self.block2 = ResBlock_model2(32, 32)
        self.block3 = ResBlock_model2(32, 64)
        self.block4 = ResBlock_model2(64, 64)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(64, 512)

    def forward(self, x):
        """
        Forward pass to obtain triplet embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: L2-normalized embedding.
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x


if __name__ == '__main__':
    input_size = (1, 410)
    model = model_2(8).cuda()
    summary(model, input_size)
