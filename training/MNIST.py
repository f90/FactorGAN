from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms

from models.MNISTClassifier import MNISTModel


def train(model, device, train_loader, optimizer, epoch):
    '''
    Train MNIST model for one epoch
    :param model: MNIST model
    :param device: Device to use
    :param train_loader: Training dataset loader
    :param optimizer: Optimiser to use
    :param epoch: Current epoch index
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    '''
    Test MNIST classifier, prints out results into standard output
    :param model: Classifier model
    :param device: Device to use
    :param test_loader: Test dataset loader
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(opt):
    use_cuda = opt.cuda and torch.cuda.is_available()

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                              transform=transforms.Compose([transforms.ToTensor()])),
                                               batch_size=opt.batchSize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batchSize, shuffle=True, **kwargs)


    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Check if we saved the model before already, in that case, just load that!
    MODEL_NAME = "mnist_classifier_model"
    if os.path.exists(MODEL_NAME):
        print("Found pre-trained MNIST classifier, loading from " + MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_NAME))
        return model

    # Train model for a certain number of epochs
    NUM_EPOCHS = 4
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    torch.save(model.state_dict(), MODEL_NAME)
    return model

if __name__ == '__main__':
    main()