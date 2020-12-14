"""Train the model"""

import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.nn.utils.prune as prune

import models.nets as nets
import models.vgg as vgg
import models.data_loaders as data_loaders
from evaluate import evaluate
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    default='mnist',
                    choices=['cifar10', 'mnist'],
                    help="Choose dataset (cifar10 or mnist)")
parser.add_argument('--model', 
                    default='fc',
                    choices=['lenet5', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'mlp', 'fc', 'conv4'],
                    help="Choose model (lenet5, vgg[11, 13, 16, 19], or mlp")
parser.add_argument('--model_dir', 
                    default='experiments/mnist_fc',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', 
                    default=None,
                    choices=['best', 'last'],
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")


def train(pruner, model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()
    pruner.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # with tqdm(total=len(dataloader), ncols=80, disable=True) as t:
    with tqdm(disable=False) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # prune model
            pruner(model)

            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            flat_model_weights = utils.flatten_model_weights(model)
            flat_masks = pruner.get_flat_masks()
            loss = loss_fn(output_batch, labels_batch, flat_masks, flat_model_weights)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # print('pruner.masks_before_sigmoid[3].grad', pruner.masks_before_sigmoid[3].grad)
            # module_list = list(model.named_modules())
            # name, mod = module_list[2]
            # print('mod.weight.grad', mod.weight.grad)
            # print('mod.weight_mask.grad', mod.weight_mask.grad)
            # print('mod.weight_orig.grad', mod.weight_orig.grad)
            # exit()

            # performs updates using calculated gradients
            optimizer.step()

            # Do not actually prune model, so we should undo pruning after each batch
            nets.undo_pruning(model)
            

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch, pruner)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean



def train_and_evaluate(pruner, model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params, 
                        model_dir, path_to_pruners, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        path_to_pruners: (string) directory containing the saved pruners.
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, pruner, optimizer)

    best_val_acc = 0.0

    train_accuracy_history = [0 for x in range(params.num_epochs)]
    eval_accuracy_history = [0 for x in range(params.num_epochs)]
    non_zero_mask_percentage_history = [0 for x in range(params.num_epochs)]

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_val_metrics = train(pruner, model, optimizer, loss_fn, train_dataloader, metrics, params)

        # # Evaluate for one epoch on validation set
        eval_val_metrics = evaluate(pruner, model, loss_fn, val_dataloader, metrics, params)

        train_accuracy_history[epoch] = train_val_metrics['accuracy']
        eval_accuracy_history[epoch] = eval_val_metrics['accuracy']
        non_zero_mask_percentage_history[epoch] = eval_val_metrics['non_zero_mask_percentage']

        pickle.dump(train_accuracy_history, open(os.path.join(model_dir, 'train_accuracy_history.p'), 'wb'))
        pickle.dump(eval_accuracy_history, open(os.path.join(model_dir, 'eval_accuracy_history.p'), 'wb'))
        pickle.dump(non_zero_mask_percentage_history, open(os.path.join(model_dir, 'non_zero_mask_percentage_history.p'), 'wb'))

        # update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        val_acc = eval_val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'pruner_dict': pruner.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(eval_val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(eval_val_metrics, last_json_path)
        pickle.dump(pruner, open(os.path.join(path_to_pruners, 'pruner_'+str(epoch)+'.p'), 'wb'))
    # save a history of the metrics
    pickle.dump(train_accuracy_history, open(os.path.join(model_dir, 'train_accuracy_history.p'), 'wb'))
    pickle.dump(eval_accuracy_history, open(os.path.join(model_dir, 'eval_accuracy_history.p'), 'wb'))
    pickle.dump(non_zero_mask_percentage_history, open(os.path.join(model_dir, 'non_zero_mask_percentage_history.p'), 'wb'))


def main():
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    path_to_pruners = args.model_dir + '/pruners'
    if not os.path.exists(path_to_pruners):
        os.mkdir(path_to_pruners)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(138)
    if params.cuda:
        torch.cuda.manual_seed(138)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    if args.dataset == 'cifar10':
        dataloaders = data_loaders.cifar10_dataloader(params.batch_size)
    else:
        dataloaders = data_loaders.mnist_dataloader(params.batch_size)
    train_dl = dataloaders['train']
    val_dl = dataloaders['test'] # !!!!!!!!!!!!!

    logging.info("- done.")

    # Define model
    if args.model == 'lenet5':
        model = nets.LeNet5(params).cuda() if params.cuda else nets.LeNet5(params)
    elif args.model[:3] == 'vgg':
        model = vgg.VGG(args.model, params).cuda() if params.cuda else vgg.VGG(args.model, params)
    elif args.model == 'conv4':
        model = nets.Conv4(params).cuda() if params.cuda else nets.Conv4(params)
    elif args.model == 'fc':
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

    # Train the model
    logging.info(args)
    logging.info(params)
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(pruner, model, train_dl, val_dl, optimizer, scheduler, loss_fn, metrics, params,
                        args.model_dir, path_to_pruners, args.restore_file)




if __name__ == '__main__':
    main()
