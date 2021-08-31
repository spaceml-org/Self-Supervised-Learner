from torch.nn import functional as F
from torch import nn
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50


class miniCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(16, 16))
        self.conv4 = nn.Conv2d(
            in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=2
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, self.output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.adaptive_pool(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return [x]
