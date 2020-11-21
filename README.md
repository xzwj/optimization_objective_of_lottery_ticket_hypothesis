# Optimization Objective of Lottery Ticket Hypothesis

## Code Layout
```
.
├── evaluate.py
├── experiments/
├── data/
│   ├── cifar-10-batches-py/
│   └── MNIST/
├── models/
│   ├── data_loaders.py
│   ├── nets.py
│   └── vgg.py
├── requirements.txt
├── train.py
└── utils.py
```

-	train.py: contains main training loop
-	utils.py: utility functions
-	evaluate.py: contains main evaluation loop
-	data/: store datasets
-	models/data_loaders.py: data loaders for each dataset
-	models/vgg.py: VGG11/13/16/19
-	models/nets.py: MLP, LeNet5, loss and evaluation metrics
-	experiments/: store hyperparameters, model weight parameters, checkpoint and training log of each experiments

## Requirements
Create a conda environment and install requirements using pip:
```python
>>> conda create -n lth python=3.7
>>> source activate lth
>>> pip install -r requirements.txt
```

## How to Run
Train a model with the specified hyperparameters:
```
>>> python train.py --model {model name} --dataset {dataset name} --model_dir {hyperparameter directory}
```

For example, using LeNet5 and MNIST to train a model with the hyperparameters in experiments/mnist_lenet5/params.json:
```
>>> python train.py --model lenet5 --dataset mnist --model_dir experiments/mnist_lenet5
```
It will automatically download the dataset and puts it in “data” directory if the dataset is not downloaded. During the training loop, best model weight parameters, last model weight parameters, checkpoint, and training log will be saved in experiments/mnist_lenet5.
