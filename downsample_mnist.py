import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# Define a transformation to downsample images to 7x7
transform = transforms.Compose([
    transforms.Resize((7, 7)),
    transforms.ToTensor()
])

# Load the MNIST dataset with the transformation
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Directory to save the downsampled dataset
output_dir = './downsampled_mnist'
os.makedirs(output_dir, exist_ok=True)

# Function to save the dataset

def save_dataset(loader, output_path):
    data = []
    labels = []
    # indices = []
    # for images, targets, idx in loader:
    for images, targets in loader:
        data.append(images)
        labels.append(targets)
        # indices.append(idx)
    data = torch.cat(data)
    labels = torch.cat(labels)
    # indices = torch.cat(indices)
    torch.save((data, labels, indices), output_path)

# Save the downsampled datasets
save_dataset(train_loader, os.path.join(output_dir, 'mnist_train_7x7.pt'))
save_dataset(test_loader, os.path.join(output_dir, 'mnist_test_7x7.pt'))

print('Downsampled MNIST dataset saved to', output_dir) 


