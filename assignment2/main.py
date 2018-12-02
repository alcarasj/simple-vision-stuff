from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import matplotlib as mpl
mpl.use('TkAgg')
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models import CNNMNISTExample, CNN6Layer3x3, CNN4Layer5x5

parser = argparse.ArgumentParser(description='Convulutional neural network model for Fashion-MNIST.')
parser.add_argument('-e', '--epochs', dest='epochs', help='The number of epochs to train the model. Default is 20.', type=int, default=20)
parser.add_argument('-b', '--batch-size', dest='batch_size', help='The batch size to propagate through the model. Default is 128.', type=int, default=128)
parser.add_argument('-l', '--learning-rate', dest='learning_rate', help='The learning rate. Default is 0.01.', type=float, default=0.01)
parser.add_argument('-m', '--momentum', dest='momentum', help='The momentum. Default is 0.25.', type=float, default=0.25)
parser.add_argument('-o', '--output-path', dest='output_path', help='Output path for saving the model. Default is ./model.pt', type=str, default='./model.pt')
parser.add_argument('-c', '--cnn', dest='cnn', help='The CNN to use. 1 for MNIST example, 2 for 6-layer 3x3 CNN, or 3 for 4-layer 5x5 CNN. Default is 3.', type=int, default=3)
args = parser.parse_args()

OUTPUT_PATH = args.output_path
if args.cnn == 1:
    CNN = CNNMNISTExample
elif args.cnn == 2:
    CNN = CNN6Layer3x3
else:
    CNN = CNN4Layer5x5

# Hyper-parameters.
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum
LOG_INTERVAL = 20

def save_predictions(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
    with open('predictions.txt', 'a') as out_file:
        numpy.savetxt(out_file, output)

def plot_data(data, label, text):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title(text + ': {}'.format(label[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
	
def predict_batch(model, device, test_loader):
    examples = enumerate(test_loader)
    model.eval()
    with torch.no_grad():
        batch_idx, (data, target) = next(examples)
        data, target = data.to(device), target.to(device)
        output = model(data)
        # Get the index of the max log-probability.
        pred = output.cpu().data.max(1, keepdim=True)[1] 
        pred = pred.numpy()
    return data, target, pred

def plot_graph(train_x, train_y, test_x, test_y, ylabel=''):
    fig = plt.figure()
    plt.plot(train_x, train_y, color='blue')
    plt.plot(test_x, test_y, color='red')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def train(model, device, train_loader, optimizer, epoch, losses=[], counter=[], errors=[]):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            losses.append(loss.item())
            counter.append((batch_idx*BATCH_SIZE) + ((epoch-1)*len(train_loader.dataset)))
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    errors.append(100. * (1 - correct / len(train_loader.dataset)))

def test(model, device, test_loader, losses=[], errors=[]):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    losses.append(test_loss)
    errors.append(100. *  (1 - correct / len(test_loader.dataset)))
  
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('\nCUDA is enabled.' if use_cuda else 'Not using CUDA.')
    print('Selected CNN is %s.' % CNN().name)
    print('\nHYPER-PARAMETERS')
    print('Epochs: %i' % EPOCHS)
    print('Batch size: %i' % BATCH_SIZE)
    print('Learning rate: %f' % LEARNING_RATE)
    print('Momentum: %f' % MOMENTUM)

    # Data transformation.
    train_data = datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    test_data = datasets.FashionMNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1325,), (0.3105,))
                   ]))

    # Data loaders.
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

	# Extract and plot random samples of data.
    examples = enumerate(test_loader)
    batch_idx, (data, target) = next(examples)
	
    # Model creation.
    model = CNN().to(device)
    # Optimizer creation.
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Lists for saving history.
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(EPOCHS + 1)]
    train_errors = []
    test_errors = []
    error_counter = [i*len(train_loader.dataset) for i in range(EPOCHS)]

    # Test of randomly initialized model.
    test(model, device, test_loader, losses=test_losses)

    # Global training and testing loop.
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, losses=train_losses, counter=train_counter, errors=train_errors)
        test(model, device, test_loader, losses=test_losses, errors=test_errors)
       
    torch.save(model.state_dict(), OUTPUT_PATH)
    save_predictions(model, device, test_loader)

    # Plotting training history.
    plot_graph(train_counter, train_losses, test_counter, test_losses, ylabel='Negative log likelihood loss')
    plot_graph(error_counter, train_errors, error_counter, test_errors, ylabel='Error (%)')
	
if __name__ == '__main__':
    main()
