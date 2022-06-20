import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


def visualizer(train_loader,id):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    # get one image from the batch
    img = np.squeeze(images[id])

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()



class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)

        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim, output_size)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        # flatten image
        x = x.view(-1, 28*28)
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer
        out = self.fc4(x)

        return x


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)

        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim * 4, output_size)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer with tanh applied
        out = F.tanh(self.fc4(x))

        return x


# Calculate losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)  # real labels = 1

    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

if __name__ == '__main__':


    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # get the training datasets
    train_data = datasets.MNIST(root='data', train=True,
                                download=True, transform=transform)

    # prepare data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers)
    #visualization of sample image
    #visualizer(train_loader,2)



    # Discriminator hyperparams

    # Size of input image to discriminator (28*28)
    input_size = 784
    # Size of discriminator output (real or fake)
    d_output_size = 1
    # Size of last hidden layer in the discriminator
    d_hidden_size = 32

    # Generator hyperparams

    # Size of latent vector to give to generator
    z_size = 100
    # Size of discriminator output (generated image)
    g_output_size = 784
    # Size of first hidden layer in the generator
    g_hidden_size = 32

    # instantiate discriminator and generator
    D = Discriminator(input_size, d_hidden_size, d_output_size)
    G = Generator(z_size, g_hidden_size, g_output_size)

    # check that they are as you expect
    print(D)
    print()
    print(G)

    import torch.optim as optim

    # Optimizers
    lr = 0.002

    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)