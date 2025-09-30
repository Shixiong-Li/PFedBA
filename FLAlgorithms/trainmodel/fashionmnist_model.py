import torch.nn as nn
import torch.nn.functional as F
from .simple import SimpleNet

class FMnistNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(FMnistNet, self).__init__(f'{name}_Simple', created_time)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_feature(self, x):
        # 前向传播直到第二个最大池化层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x  # 返回特征图，形状为 [batch_size, 50, 4, 4]

    def get_feature_list(self, x):
        feature_list = []

        x = self.conv1(x)
        x = F.relu(x)
        feature_list.append(x.clone().detach())  # conv1 output

        x = F.max_pool2d(x, 2, 2)
        feature_list.append(x.clone().detach())  # after first maxpool

        x = self.conv2(x)
        x = F.relu(x)
        feature_list.append(x.clone().detach())  # conv2 output

        x = F.max_pool2d(x, 2, 2)
        feature_list.append(x.clone().detach())  # after second maxpool

        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = F.relu(x)
        feature_list.append(x.clone().detach())  # fc1 output

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        feature_list.append(x.clone().detach())  # final output

        return feature_list


if __name__ == '__main__':
    model=FMnistNet()
    print(model)




