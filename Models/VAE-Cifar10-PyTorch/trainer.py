import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="2" 
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# from model import BN_VAE as vae
from model import LinearVAE as vae
from utils import BCE_KLD_Loss as Loss

parser = argparse.ArgumentParser(description='VAE MNIST Implementation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--zdims', type=int, default=512, metavar='N', help='number of dimensions of the latent vector (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

# where to load cifar10 data
cifar10_path = '../VAE-PyTorch/data'

args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(cifar10_path, train=True, download=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# load the testing data (MNIST Data Set)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(cifar10_path, train=False, download=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = vae().to(device)
# print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch, print_loss=False):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = Loss(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if print_loss:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, 
                    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data))
                )

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, 
          train_loss / len(train_loader.dataset))
    )

def test(epoch):

    if not os.path.exists(os.path.join(os.getcwd(), 'Results')):
        os.mkdir(os.path.join(os.getcwd(), 'Results'))

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += Loss(recon_batch, data, mu, logvar).item()

            if i == 0:
                n = min(data.size(0), 8)

                comparison = torch.cat(
                	[data[:n],
                	recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
                save_image(comparison.data.cpu(),
                           'Results/epoch_' + str(epoch) + '.png', 
                           nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test in Epoch %d'%(epoch))
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % 10 == 0 or epoch == args.epochs:
        	test(epoch)
