import logging

import torch
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO)


def get_train_loader(data_dir, batch_size, cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=True, download=False,
                         transform=transforms.Compose(
                           [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                         ),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader


def get_test_loader(data_dir, batch_size, cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=False,
                         transform=transforms.Compose(
                           [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                       ),
        batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader


def train(model, train_loader, epoch, cuda, optimizer, criterion, log_interval):
    model.train()

    # Get data
    for batch_idx, (data, label) in enumerate(train_loader):
        if cuda:
            data, label = data.cuda(), label.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Print statistics
        if batch_idx % log_interval == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item())
            )

    print('Finished Training')


def test(experiment, model, criterion, test_loader, cuda):
    model.eval()

    correct = 0

    with torch.no_grad():
        # Get data
        for data, label in test_loader:
            if cuda:
                data, label = data.cuda(), label.cuda()

            # Run data over the network
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()

    logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / len(test_loader.dataset)))
