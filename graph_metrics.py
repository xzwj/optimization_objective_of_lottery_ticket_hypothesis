import pickle
import matplotlib.pyplot as plt
import os
import utils
import models.nets as nets

# change the dataset and model_name
# dataset = 'mnist'
# model = 'fc'

dataset = 'mnist'
model = 'lenet5'

# dataset = 'cifar10'
# model = 'conv4'
#
# dataset = 'cifar10'
# model = 'lenet5'

model_dir = 'experiments/' + dataset + '_' + model
json_path = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

path_to_pruners = model_dir + '/pruners'
pruner_by_epoch = []
for epoch in range(params.num_epochs):
    path_to_pruner = os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.pt')
    pruner = pickle.load(open(os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.p'), 'rb'))
    pruner_by_epoch.append(pruner)

train_accuracy_history = pickle.load(open(os.path.join(model_dir, 'train_accuracy_history.p'), 'rb'))
eval_accuracy_history = pickle.load(open(os.path.join(model_dir, 'eval_accuracy_history.p'), 'rb'))
non_zero_mask_percentage_history = pickle.load(open(os.path.join(model_dir, 'non_zero_mask_percentage_history.p'), 'rb'))

print('train_accuracy_history')
print(train_accuracy_history)
print('eval_accuracy_history')
print(eval_accuracy_history)
print('non_zero_mask_percentage_history')
print(non_zero_mask_percentage_history)

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
plt.title('Percentage and mean of remaining weights ({} {})'.format(dataset.upper(), model.upper()))
plt.plot(non_zero_mask_percentage_history, color='black', linestyle='dashed')
plt.ylim(-0.05, 1.05)
plt.xlabel('Epoch')
plt.ylabel('Remaining weights %')
# plt.legend(['Remaining weights %', 'mean of remaining weights', 'mean of all weights'])
# plt.legend(['Remaining weights %'])
fig_2.savefig(os.path.join(path_to_graphs, 'non_zero_mask_percentage.png'))
plt.close(fig_2)
