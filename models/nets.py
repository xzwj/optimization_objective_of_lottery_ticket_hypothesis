import torch.nn as nn
import torch.nn.functional as F
# import torch
import numpy as np


class LeNet5(nn.Module):
    def __init__(self, params):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(params.input_channel, 6, 5), # Params: (input_channel, output_channel, kernel_size)
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # Params: (kernel_size, stride)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
        self.fc1 = nn.Linear(16 * (params.input_h//4-3) * (params.input_w//4-3), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, params.num_class)

        self.masks = []

        # TODO: 
        # 2. get the shape of a
        # 3. initial a
        # 4. z = torch.sigmoid(a)

    def forward(self, x):
        # TODO:
        # 5. theta * z

        # Input: (batch_size, input_channel, input_h, input_w)
        # After Conv2D: (batch_size, 6, input_h - 4, input_w - 4)
        # After MaxPool2D: (batch_size, 6, (input_h - 4) // 2, (input_w - 4) // 2)
        x = self.conv1(x)

        # After Conv2D: (batch_size, 16, input_h // 2 - 6, input_w // 2 - 6) 
        # After MaxPool2D: (batch_size, 16, (input_h // 2 - 6) // 2, (input_w // 2 - 6) // 2)
        x = self.conv2(x)

        # Flat 2-dim features to 1-dim
        x = x.view(x.size(0), -1) # (batch_size, 16 * (input_h // 4 - 3) * (input_w // 4 - 3))
        x = F.relu(self.fc1(x)) # (batch_size, 120)
        x = F.relu(self.fc2(x)) # (batch_size, 84)
        # Finally we have 10 class
        x = self.fc3(x) # (batch_size, num_class)

        # Apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(params.input_channel * params.input_h * params.input_w, params.fc1_out),
                nn.ReLU(),
                nn.Dropout(params.dropout1),
            )
        self.fc2 = nn.Sequential(
                nn.Linear(params.fc1_out, params.fc2_out),
                nn.ReLU(),
                nn.Dropout(params.dropout2),
            )
        self.fc3 = nn.Linear(params.fc2_out, params.num_class)

    def forward(self, x):
        # Flat 2-dim features to 1-dim
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

        


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) (batch_size, num_class) - log softmax output of the model
        labels: (np.ndarray) dimension (batch_size), where each element is a value in [0, 1, 2, ..., num_class-1]
    Returns: (float) accuracy in [0,1]
    """
    # print('outputs', outputs)
    # print('labels', labels)
    outputs = np.argmax(outputs, axis=1)
    # print('outputs', outputs)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}




class Loss(nn.Module):
    """docstring for Loss"""
    def __init__(self, params, flatten_model_orig_weights):
        super(Loss, self).__init__()

        self.lambda1 = params.lambda1
        self.lambda2 = params.lambda2

        self.flat_model_orig_weights = flat_model_orig_weights

        self.loss1 = nn.NLLLoss()
        # self.loss2 = 
        # TODO:
        # 6. define loss2 and loss3

    def forward(self, outputs, labels, masks, flat_model_weights):
        return self.loss1(outputs, labels) + \
                self.lambda1 * self.loss2(masks) + \
                self.lambda2 * self.loss3(masks, flat_model_weights)


        




if __name__ == '__main__':
    # Test for class `LeNet5`
    import torch
    import sys 
    sys.path.append(".") 
    import utils

    params = utils.Params('./experiments/cifar10_lenet5/params.json')
    model = LeNet5(params)
    print(model)
    x = torch.randn(2,3,32,32)
    print(x)
    y = model(x)
    print(y)
    print(y.size())
    


