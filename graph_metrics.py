import pickle
import matplotlib.pyplot as plt
import os
import utils
import graph_metrics_utils
import models.nets as nets

# change the dataset and model_name
# dataset = 'mnist'
# model = 'fc'

# dataset = 'mnist'
# model = 'lenet5'

dataset = 'cifar10'
model = 'conv4'

# dataset = 'cifar10'
# model = 'lenet5'

model_dir = 'experiments/' + dataset + '_' + model
json_path = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

# load the pruners
path_to_pruners = model_dir + '/pruners'
pruner_by_epoch = []
for epoch in range(params.num_epochs):
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
last_epoch_flat_masks = pruner_by_epoch[-1].get_flat_masks().detach().numpy()
last_epoch_non_zero_masks = last_epoch_flat_masks[last_epoch_flat_masks >= threshold]

# initialize path to save graphs
path_to_graphs = model_dir + '/graphs'
if not os.path.exists(path_to_graphs):
    os.mkdir(path_to_graphs)

# plot accuracy history
fig_1 = plt.figure(1)
plt.title('Training and testing accuracy ({} {})'.format(dataset.upper(), model.upper()))
plt.plot(train_accuracy_history, color='blue')
plt.plot(eval_accuracy_history, color='red')
plt.ylim(-0.05, 1.05)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'Test'])
fig_1.savefig(os.path.join(path_to_graphs, 'accuracy.png'))
plt.close(fig_1)

# plot mask history
fig_2 = plt.figure(2)
plt.title('Percent and mean of masks ({} {})'.format(dataset.upper(), model.upper()))
plt.plot(non_zero_mask_percentage_history, color='black', linestyle='dashed')
plt.plot(mean_of_all_masks, color='blue')
plt.plot(mean_of_non_zero_masks, color='red')
plt.ylim(-0.05, 1.05)
plt.xlabel('epoch')
plt.ylabel('% and mean of masks')
plt.legend(['% of remaining masks', 'mean of all masks', 'mean of non-zero masks'])
fig_2.savefig(os.path.join(path_to_graphs, 'percent_and_mean_of_masks.png'))
plt.close(fig_2)

# plot histogram of masks
num_bins = 20
# bins_list = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fig_3 = plt.figure(3)
plt.title('Histogram of mask values ({} {})'.format(dataset.upper(), model.upper()))
plt.hist(last_epoch_flat_masks, bins=num_bins, histtype='bar')
plt.xlabel('mask value')
plt.ylabel('num of masks')
fig_3.savefig(os.path.join(path_to_graphs, 'histogram_of_masks.png'))
plt.close(fig_3)

# plot histogram of non-zero masks
num_bins = 20
# bins_list = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fig_4 = plt.figure(4)
plt.title('Histogram of non-zero mask values ({} {})'.format(dataset.upper(), model.upper()))
plt.hist(last_epoch_non_zero_masks, bins=num_bins, histtype='bar')
plt.xlabel('mask value')
plt.ylabel('num of masks')
fig_4.savefig(os.path.join(path_to_graphs, 'histogram_of_non_zero_masks.png'))
plt.close(fig_4)