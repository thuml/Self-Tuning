import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, inputs, class_num):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(inputs, class_num)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier_layer(x)