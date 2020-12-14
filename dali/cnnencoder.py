import torch
from torch.nn import functional as F
from torch import nn

class miniCNN(nn.Module):

  def __init__(self):
    super().__init__()

    # merced images are (3, 256, 256) (channels, width, height)
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 0)
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 2, padding = 0)
    self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = 3, stride = 2, padding = 0)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64*3*3, 128)

    self.train_acc = Accuracy()
    self.val_acc = Accuracy(compute_on_step=False)
    self.test_acc = Accuracy(compute_on_step=False)


  def forward(self, x):
      # print(x.shape)
      x = F.relu(self.conv1(x))
      # print(x.shape)
      x = self.pool(x)
      # print(x.shape)
      x = F.relu(self.conv2(x))
      # print(x.shape)
      x = self.pool(x)
      # print(x.shape)
      x = F.relu(self.conv3(x))
      x = self.pool(x)
      # print(x.shape)
      x = F.relu(self.conv4(x))
      x = self.pool(x)
      # print(x.shape)
      x = x.view(-1 , 64*3*3)
      # print(x.shape)

      return [x]
