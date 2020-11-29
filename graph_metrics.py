import pickle
import matplotlib.pyplot as plt
import os


# change the dataset and model_name
dataset = 'mnist'
model_name = 'fc'

model_dir = 'experiments/' + dataset + '_' + model_name

path_to_graphs = model_dir + '/graphs'
if not os.path.exists(path_to_graphs):
    os.mkdir(path_to_graphs)

train_accuracy_history = pickle.load(open(os.path.join(model_dir, 'train_accuracy_history.p'), 'rb'))
train_zero_mask_percentage_history = pickle.load(open(os.path.join(model_dir, 'train_non_zero_mask_percentage_history.p'), 'rb'))
eval_accuracy_history = pickle.load(open(os.path.join(model_dir, 'eval_accuracy_history.p'), 'rb'))
eval_zero_mask_percentage_history = pickle.load(open(os.path.join(model_dir, 'eval_non_zero_mask_percentage_history.p'), 'rb'))

print(train_zero_mask_percentage_history)
print(eval_zero_mask_percentage_history)

# plot training history
fig_1 = plt.figure(1)
plt.title('Training and testing accuracy ({} {})'.format(dataset.upper(), model_name.upper()))
plt.plot(train_accuracy_history, color='blue', label='accuracy')
plt.plot(eval_accuracy_history, color='red', label='pruned weights')
plt.ylim(-0.05, 1.05)
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'Test'])
fig_1.savefig(os.path.join(path_to_graphs, 'accuracy.png'))
plt.close(fig_1)

# plot testing history
fig_2 = plt.figure(1)
plt.title('Training and testing percentage of pruned weights ({} {})'.format(dataset.upper(), model_name.upper()))
plt.plot(train_zero_mask_percentage_history, color='blue', label='accuracy')
plt.plot(eval_zero_mask_percentage_history, color='red', label='pruned weights')
plt.ylim(-0.05, 1.05)
plt.xlabel('Epoch')
plt.ylabel('Pruned weights %')
plt.legend(['Train', 'Test'])
fig_2.savefig(os.path.join(path_to_graphs, 'zero_mask_percentage.png'))
plt.close(fig_2)
