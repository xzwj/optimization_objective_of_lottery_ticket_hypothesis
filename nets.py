import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.utils.prune as prune


class Pruner(nn.Module):
    def __init__(self, model, mask_init):
        super(Pruner, self).__init__()
        self.masks_before_sigmoid = self._init_masks_before_sigmoid(model, mask_init)
        
    def forward(self, model):
        i = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                weight_mask = torch.sigmoid(self.masks_before_sigmoid[i])
                prune.custom_from_mask(mod, 'weight', weight_mask)

                bias_mask = torch.sigmoid(self.masks_before_sigmoid[i+1])
                prune.custom_from_mask(mod, 'bias', bias_mask)

                i += 2

    def _init_masks_before_sigmoid(self, model, mask_init):
        masks_before_sigmoid = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                masks_before_sigmoid.append(nn.Parameter(torch.full_like(mod.weight.data, mask_init), requires_grad=True))
                masks_before_sigmoid.append(nn.Parameter(torch.full_like(mod.bias.data, mask_init), requires_grad=True))
        return nn.ParameterList(masks_before_sigmoid)

    def get_flat_masks(self):
        return torch.cat(tuple(torch.flatten(torch.sigmoid(mbs)) for mbs in self.masks_before_sigmoid))
        


def undo_pruning(model): 
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            mod.weight_mask.fill_(1.0)
            mod.bias_mask.fill_(1.0)

            prune.remove(mod, 'weight')
            prune.remove(mod, 'bias')

        


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

    def forward(self, x):
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

        return F.log_softmax(x, dim=1)


class Conv4(nn.Module):
    def __init__(self, params):
        '''

        Args: CIFAR10 Only
            params:
        '''
        super(Conv4, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(params.input_channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        # Input: (batch_size, input_channel, input_h, input_w)
        # After Conv2D: (batch_size, 6, input_h - 4, input_w - 4)
        # After MaxPool2D: (batch_size, 6, (input_h - 4) // 2, (input_w - 4) // 2)
        x = self.model(x)
        # Flat 2-dim features to 1-dim
        x = x.view(x.shape[0], -1) # (batch_size, 16 * (input_h // 4 - 3) * (input_w // 4 - 3))
        # Finally we have 10 class
        x = self.fc(x) # (batch_size, num_class)

        return F.log_softmax(x, dim=1)


class fc(nn.Module):
    def __init__(self, params):
        super(fc, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(params.input_channel * params.input_h * params.input_w, 300),
                nn.ReLU(),
            )
        self.fc2 = nn.Sequential(
                nn.Linear(300, 100),
                nn.ReLU(),
            )
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        # Flat 2-dim features to 1-dim
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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
    def __init__(self, params, flat_model_orig_weights):
        """
        Args:
            flat_model_orig_weights: size (total_num_model_params)
        """
        super(Loss, self).__init__()

        self.lambda1 = params.lambda1
        self.lambda2 = params.lambda2

        self.flat_model_orig_weights = flat_model_orig_weights
        self.len_flat_model_weight = flat_model_orig_weights.shape[0]

        self.loss1 = nn.NLLLoss()   

    def forward(self, outputs, labels, flat_masks, flat_model_weights):
        return self.loss1(outputs, labels) + \
                self.lambda1 * self.loss2(flat_masks) + \
                self.lambda2 * self.loss3(flat_masks, flat_model_weights)

    def loss2(self, flat_masks):
        l1_norm = torch.norm(flat_masks, p=1)
        return l1_norm / self.len_flat_model_weight

    def loss3(self, flat_masks, flat_model_weights):
        l2_norm = torch.norm(flat_model_weights * flat_masks - self.flat_model_orig_weights * flat_masks, p=2)
        return l2_norm / self.len_flat_model_weight


    


        




if __name__ == '__main__':
    import torch
    import sys 
    sys.path.append(".") 
    import utils

    # # ================== Test for class `LeNet5` ==================
    # params = utils.Params('./experiments/mnist_lenet5/params.json')
    # model = LeNet5(params)
    # print(model)

    # # ================== Test for class `Loss` ==================
    # flat_model_orig_weights = utils.flatten_model_weights(model)
    # loss_fn = Loss(params, flat_model_orig_weights)

    # x = torch.randn(2,1,28,28)
    # y = model(x)
    # labels = torch.tensor([1, 0])
    # flat_model_weights = torch.randn(44426)
    # flat_masks = torch.randn(44426)
    # loss = loss_fn(y, labels, flat_masks, flat_model_weights)
    # print('loss', loss)

    # # ================== Test for class `Pruner.__init__()` ==================
    # module_list = list(model.named_modules())
    # print('module_list', module_list)
    # print('module_list[2]', module_list[2])
    # name, mod = module_list[2]
    # # print('mod.weight', mod.weight)
    # # print('mod.weight.shape', mod.weight.shape)
    # # print('mod.weight.data', mod.weight.data)
    # # print('mod.weight.data.shape', mod.weight.data.shape)
    # # print('mod.weight.data.dtype', mod.weight.data.dtype) # torch.float32
    # # print(torch.tensor([5.0, 5.0]).dtype) # torch.float32

    # pruner = Pruner(model, params.mask_init)
    # print('len(pruner.masks_before_sigmoid)', len(pruner.masks_before_sigmoid))
    # for p in pruner.masks_before_sigmoid:
    #     print(p.data.shape)

    # # ================== Test for class `Pruner.get_flat_masks() ==================
    # # print(list(model.parameters()))
    # print("============")
    # for p in model.parameters():
    #     print(p.data.shape)

    # flat_masks = pruner.get_flat_masks()
    # print('flat_masks.shape', flat_masks.shape)

    # ================== Test for `undo_pruning()` ==================
    model = nn.Linear(10, 1)
    print(list(model.named_parameters()))
    print(list(model.named_buffers()))
    print('=======1========')

    prune.random_unstructured(model, name="weight", amount=0.3)
    prune.random_unstructured(model, name="bias", amount=0.3)
    print(list(model.named_parameters()))
    print(list(model.named_buffers()))
    print('=======2========')

    undo_pruning(model)
    print(list(model.named_parameters()))
    print(list(model.named_buffers()))








    


