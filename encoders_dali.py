import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.metrics import Accuracy
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50

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



def load_encoder(encoder_name, kwargs):
    if encoder_name == 'minicnn':
        model, embedding_size = miniCNN(), 576
    elif encoder_name == 'resnet18':
        model, embedding_size = resnet18(pretrained=kwargs['pretrained'], first_conv=True, maxpool1=True, return_all_feature_maps=False), 512
    elif encoder_name == 'resnet50':
        model, embedding_size = resnet50(pretrained=kwargs['pretrained'], first_conv=True, maxpool1=True, return_all_feature_maps=False), 2048
    else:
        raise Exception('Encoder specified is not supported')

    if 'MODEL_PATH' in kwargs and kwargs['MODEL_PATH'] is not None:
        print('Loading Model from save path')
        model.load_state_dict(torch.load(kwargs['MODEL_PATH']))

    return model, embedding_size
  
def get_size(encoder_name, kwargs):
    if encoder_name == 'minicnn':
        return 576
    elif encoder_name == 'resnet18':
        return 512
    elif encoder_name == 'resnet50':
        return 2048
    else:
        raise Exception('Encoder specified is not supported')
