"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import models.nets as nets
import models.data_loaders as data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    default='mnist',
                    choices=['cifar10', 'mnist'],
                    help="Choose dataset (cifar10 or mnist)")
parser.add_argument('--model', 
                    default='lenet5',
                    choices=['lenet5', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'mlp'],
                    help="Choose model (lenet5, vgg[11, 13, 16, 19], or mlp")
parser.add_argument('--model_dir', 
                    default='experiments/mnist_lenet5',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', 
                    default='best',
                    choices=['best', 'last'],
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")


def evaluate(pruner, model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    pruner.eval()

    pruner(model)
    flat_model_weights = utils.flatten_model_weights(model)
    flat_masks = pruner.get_flat_masks()

    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in dataloader:

            # move to GPU if available
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch, flat_masks, flat_model_weights)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch, pruner)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    # undo pruning
    nets.undo_pruning(model)

    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(138)
    if params.cuda:
        torch.cuda.manual_seed(138)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    if args.dataset == 'cifar10':
        dataloaders = data_loaders.cifar10_dataloader(params.batch_size)
    else:
        dataloaders = data_loaders.mnist_dataloader(params.batch_size)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    if args.model == 'lenet5':
        model = nets.LeNet5(params).cuda() if params.cuda else nets.LeNet5(params)
    elif args.model[:3] == 'vgg':
        model = vgg.VGG(args.model, params).cuda() if params.cuda else nvgg.VGG(args.model, params)
    else:
        model = nets.MLP(params).cuda() if params.cuda else nets.MLP(params)

    # Define the pruner
    pruner = nets.Pruner(model, params.mask_init)

    # fetch loss function and metrics
    flat_model_orig_weights = utils.flatten_model_weights(model)
    loss_fn = nets.Loss(params, flat_model_orig_weights)
    metrics = nets.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model, pruner)

    # Evaluate
    test_metrics = evaluate(pruner, model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)