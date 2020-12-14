import pickle
import matplotlib.pyplot as plt
import os
import utils
import graph_metrics_utils
import models.nets as nets
import numpy as np


import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import models.vgg as vgg
import models.data_loaders as data_loaders
from evaluate import evaluate


# change the dataset and model_name
# dataset = 'mnist'
# model_name = 'fc'

# dataset = 'mnist'
# model_name = 'lenet5'

# dataset = 'cifar10'
# model_name = 'conv4'

dataset = 'cifar10'
model_name = 'lenet5'

model_dir = 'experiments/' + dataset + '_' + model_name
json_path = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

num_epochs = params.num_epochs

# if model_name == 'conv4':
#     num_epochs = 2

# load the pruners
path_to_pruners = model_dir + '/pruners'
pruner_by_epoch = []
for epoch in range(num_epochs):
    path_to_pruner = os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.pt')
    pruner = pickle.load(open(os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.p'), 'rb'))
    pruner_by_epoch.append(pruner)

# load the accuracy and percent of non-zero masks
train_accuracy_history = pickle.load(open(os.path.join(model_dir, 'train_accuracy_history.p'), 'rb'))
eval_accuracy_history = pickle.load(open(os.path.join(model_dir, 'eval_accuracy_history.p'), 'rb'))
non_zero_mask_percentage_history = pickle.load(open(os.path.join(model_dir, 'non_zero_mask_percentage_history.p'), 'rb'))

print('train_accuracy_history')
print(train_accuracy_history)
print('eval_accuracy_history')
print(eval_accuracy_history)
print('non_zero_mask_percentage_history')
print(non_zero_mask_percentage_history)

# calculate more metrics
threshold = 0.001
mean_of_all_masks = graph_metrics_utils.get_mean_of_all_masks(pruner_by_epoch)
mean_of_non_zero_masks = graph_metrics_utils.get_mean_of_non_zero_masks(pruner_by_epoch)
flat_masks = pruner_by_epoch[-1].get_flat_masks()
last_epoch_flat_masks = flat_masks.detach().numpy()
last_epoch_non_zero_masks = last_epoch_flat_masks[last_epoch_flat_masks >= threshold]

# initialize path to save graphs
path_to_graphs = model_dir + '/graphs'
if not os.path.exists(path_to_graphs):
    os.mkdir(path_to_graphs)


path_to_pruners = model_dir + '/pruners'
if not os.path.exists(path_to_pruners):
    os.mkdir(path_to_pruners)

# use GPU if available
params.cuda = torch.cuda.is_available()

# Set the random seed for reproducible experiments
torch.manual_seed(138)
if params.cuda:
    torch.cuda.manual_seed(138)

# Set the logger
utils.set_logger(os.path.join(model_dir, 'train.log'))

# Create the input data pipeline

# fetch dataloaders
# if dataset == 'cifar10':
#     dataloaders = data_loaders.cifar10_dataloader(params.batch_size)
# else:
#     dataloaders = data_loaders.mnist_dataloader(params.batch_size)
# train_dl = dataloaders['train']
# val_dl = dataloaders['test']  # !!!!!!!!!!!!!

# Define model
if model_name == 'lenet5':
    model = nets.LeNet5(params).cuda() if params.cuda else nets.LeNet5(params)
elif model_name[:3] == 'vgg':
    model = vgg.VGG(model_name, params).cuda() if params.cuda else vgg.VGG(model_name, params)
elif model_name == 'conv4':
    model = nets.Conv4(params).cuda() if params.cuda else nets.Conv4(params)
elif model_name == 'fc':
    model = nets.Fc(params).cuda() if params.cuda else nets.Fc(params)
else:
    model = nets.MLP(params).cuda() if params.cuda else nets.MLP(params)

# Define mask method
pruner = nets.Pruner(model, params.mask_init)

# Define optimizer
optim_params = list(model.parameters()) + list(pruner.parameters())
if params.optim == 'sgd':
    optimizer = optim.SGD(optim_params, lr=params.lr, momentum=params.momentum,
                          weight_decay=(params.wd if params.dict.get('wd') is not None else 0.0))
else:
    optimizer = optim.Adam(optim_params, lr=params.lr,
                           weight_decay=(params.wd if params.dict.get('wd') is not None else 0.0))

if params.dict.get('lr_adjust') is not None:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params.lr_adjust, gamma=0.1)
else:
    scheduler = None

# fetch loss function and metrics
flat_model_orig_weights = utils.flatten_model_weights(model)
loss_fn = nets.Loss(params, flat_model_orig_weights)
metrics = nets.metrics

# restore_file = 'best'
restore_file = 'last'

flat_weights_1 = utils.flatten_model_weights(model).detach().numpy()
if restore_file is not None:
    restore_path = os.path.join(
        model_dir, restore_file + '.pth.tar')
    utils.load_checkpoint(restore_path, model, pruner, optimizer)

flat_weights = utils.flatten_model_weights(model)
flat_weights_2 = utils.flatten_model_weights(model).detach().numpy()

# flat_masks, last_epoch_flat_masks (np), flat_model_weights, flat_model_weights_2 (np)

# for i in range(100):
#     print(last_epoch_flat_masks[i], flat_model_weights_2[i], last_epoch_flat_masks[i] * flat_model_weights_2[i])

# last_epoch_non_zero_masks = last_epoch_flat_masks[last_epoch_flat_masks >= threshold]
last_epoch_zero_masks = last_epoch_flat_masks[last_epoch_flat_masks < threshold]
mask_size = np.size(last_epoch_flat_masks)
non_zero_size = np.size(last_epoch_non_zero_masks)
zero_size = np.size(last_epoch_zero_masks)
assert mask_size == non_zero_size + zero_size

flat_mask_times_weight = flat_masks * flat_weights  # type tensor

# the following are np arrays
flat_non_zero_mask_times_weight = flat_mask_times_weight[last_epoch_flat_masks >= threshold].detach().numpy()
flat_zero_mask_times_weight = flat_mask_times_weight[last_epoch_flat_masks < threshold].detach().numpy()

flat_non_zero_weights_1 = flat_weights_1[last_epoch_flat_masks >= threshold]
flat_zero_weights_1 = flat_weights_1[last_epoch_flat_masks < threshold]

flat_non_zero_weights_2 = flat_weights_2[last_epoch_flat_masks >= threshold]
flat_zero_weights_2 = flat_weights_2[last_epoch_flat_masks < threshold]
print('Sizes:')
print('size(flat_model_weights_2', np.size(flat_weights_2))
print('size(last_epoch_non_zero_masks', np.size(last_epoch_non_zero_masks))
print('size(flat_non_zero_model_weights_2)', np.size(flat_non_zero_weights_2))
print('np.size(flat_zero_model_weights_2)', np.size(flat_zero_weights_2))

print()
print('Means:')
print('Mask:')
print('last_epoch_zero_masks', np.mean(last_epoch_zero_masks))
print('last_epoch_non_zero_masks', np.mean(last_epoch_non_zero_masks))
print('Weights:')
print('flat_non_zero_model_weights_2', np.mean(flat_non_zero_weights_2))
print('flat_zero_model_weights_2', np.mean(flat_zero_weights_2))
print('Mask * weight:')
print('flat_non_zero_mask_times_weight', np.mean(flat_non_zero_mask_times_weight))
print('flat_zero_mask_times_weight', np.mean(flat_zero_mask_times_weight))

# plot histogram of non-zero masks
fig_5 = plt.figure(5)
plt.title('Boxplot of masks * weights ({} {})'.format(dataset.upper(), model_name.upper()))
plt.boxplot(flat_zero_mask_times_weight)
plt.xlabel('Group')
plt.ylabel('Values')
fig_5.savefig(os.path.join(path_to_graphs, 'boxplot_of_masks_times_weights.png'))
plt.close(fig_5)

qs = np.quantile(flat_zero_mask_times_weight, [0.25, 0.5, 0.75])
print(qs)
ps = graph_metrics_utils.get_percentile_of_value(flat_zero_mask_times_weight, 0)
print(np.size(flat_zero_mask_times_weight[flat_zero_mask_times_weight < -0]))
print(ps)
# print(flat_zero_mask_times_weight[flat_zero_mask_times_weight < -0.001])
print(np.size(flat_zero_mask_times_weight[flat_zero_mask_times_weight < -0.001]))
print(np.min(flat_zero_mask_times_weight))

print()
print(dataset.upper(), model_name.upper())
print('flat_zero_mask_times_weight', np.mean(flat_zero_mask_times_weight))
print('mask_size, zero_size, non_zero_size', mask_size, zero_size, non_zero_size)

print('L1 norm flat_non_zero_weights_1', np.linalg.norm(flat_non_zero_weights_1, 1))
print('L1 norm flat_non_zero_weights_2', np.linalg.norm(flat_non_zero_weights_2, 1))
print('L1 norm of non_zero_weights difference', np.linalg.norm(flat_non_zero_weights_2-flat_non_zero_weights_1, 1))
print('L1 norm flat_zero_weights_1', np.linalg.norm(flat_zero_weights_1, 1))
print('L1 norm flat_zero_weights_2', np.linalg.norm(flat_zero_weights_2, 1))
print('L1 norm of zero_weights difference', np.linalg.norm(flat_zero_weights_2-flat_zero_weights_1, 1))

print('Average L1 norm flat_non_zero_weights_1', np.linalg.norm(flat_non_zero_weights_1, 1)/non_zero_size)
print('Average L1 norm flat_non_zero_weights_2', np.linalg.norm(flat_non_zero_weights_2, 1)/non_zero_size)
print('Average L1 norm non_zero_weights difference', np.linalg.norm(flat_non_zero_weights_2-flat_non_zero_weights_1, 1)/non_zero_size)
print('Average L1 norm flat_zero_weights_1', np.linalg.norm(flat_zero_weights_1, 1)/zero_size)
print('Average L1 norm flat_zero_weights_2', np.linalg.norm(flat_zero_weights_2, 1)/zero_size)
print('Average L1 norm of zero_weights difference', np.linalg.norm(flat_zero_weights_2-flat_zero_weights_1, 1)/zero_size)


