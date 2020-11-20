import torchvision
import torchvision.transforms as transforms
import torch


def cifar10_dataloader(batch_size):
    # cifar-10官方提供的数据集是用numpy array存储的
    # 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 在构建数据集的时候指定transform，就会应用我们定义好的transform
    # root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=False, transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              transform=transform_test)

    print(cifar_train)
    print(cifar_test)

    trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    return {'train':trainloader, 'test':testloader}


def mnist_dataloader(batch_size):
    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,
    )

    mnist_test = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=torchvision.transforms.ToTensor(), 
        download=True,
    )

    print(mnist_train)
    print(mnist_test)

    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return {'train':trainloader, 'test':testloader}



if __name__ == '__main__':
    # Test function `mnist_dataloader`
    dl = mnist_dataloader(32)
    print(dl['train'])
    print(dl['test'])






