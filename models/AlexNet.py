from torch import nn
from .BasicModule import BasicModule

class AlexNet(BasicModule):
    '''
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''
    def __init__(self, num_classes = 2):
        
        super(AlexNet,self).__init__()

        self.model_name = 'alexnet'

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, Kernel_size=11, stride=4, padding=2),
            nn.Relu(inplace=True),
            nn.MaxPool2d(Kernel_size=3, stride=2),
            nn.Conv2d(64, 192, Kernel_size=5, padding=2),
            nn.Relu(inplace=True),
            nn.MaxPool2d(Kernel_size=3, stride=2),
            nn.Conv2d(192, 384, Kernel_size=3, padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(384, 256, Kernel_size=3, padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(256, 256, Kernel_size=3, padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(Kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Relu(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Relu(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x