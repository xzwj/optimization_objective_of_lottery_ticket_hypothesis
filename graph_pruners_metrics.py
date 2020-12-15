import pickle
import matplotlib.pyplot as plt
import os
import utils
import graph_metrics_utils
import models.nets as nets
import numpy as np

# change the dataset and model_name
dataset = 'mnist'
model = 'fc'

# dataset = 'mnist'
# model = 'lenet5'

# dataset = 'cifar10'
# model = 'conv4'

# dataset = 'cifar10'
# model = 'lenet5'

model_dir = 'experiments/' + dataset + '_' + model
json_path = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
num_epochs = params.num_epochs
num_epochs = 10

# load the pruners
path_to_pruners = model_dir + '/pruners'
pruner_by_epoch = []
for epoch in range(num_epochs):
    path_to_pruner = os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.pt')
    pruner = pickle.load(open(os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.p'), 'rb'))
    pruner_by_epoch.append(pruner)


# calculate more metrics
threshold = 0.001
mean_of_all_masks = graph_metrics_utils.get_mean_of_all_masks(pruner_by_epoch)
mean_of_non_zero_masks = graph_metrics_utils.get_mean_of_non_zero_masks(pruner_by_epoch)
last_epoch_flat_masks = pruner_by_epoch[-1].get_flat_masks().detach().numpy()
last_epoch_non_zero_masks = last_epoch_flat_masks[last_epoch_flat_masks >= threshold]
print(np.size(last_epoch_flat_masks))

# initialize path to save graphs
path_to_graphs = model_dir + '/graphs'
if not os.path.exists(path_to_graphs):
    os.mkdir(path_to_graphs)


# plot histogram of masks
# num_bins = 20
bins_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
fig_3 = plt.figure(3)
plt.title('Histogram of mask values after {} epochs ({} {})'.format(num_epochs, dataset.upper(), model.upper()))
plt.hist(last_epoch_flat_masks, bins=bins_list, histtype='bar')
plt.xlabel('mask value')
plt.ylabel('num of masks')
fig_3.savefig(os.path.join(path_to_graphs, 'histogram_of_masks_'+str(num_epochs)+'.png'))
plt.close(fig_3)

# plot histogram of non-zero masks
fig_4 = plt.figure(4)
plt.title('Histogram of non-zero mask values after {} epochs ({} {})'.format(num_epochs, dataset.upper(), model.upper()))
plt.hist(last_epoch_non_zero_masks, bins=bins_list, histtype='bar')
plt.xlabel('mask value')
plt.ylabel('num of masks')
fig_4.savefig(os.path.join(path_to_graphs, 'histogram_of_non_zero_masks_'+str(num_epochs)+'.png'))
plt.close(fig_4)