from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),               
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    return transform

def load_MNIST_dataset(batch_size: int):
    transform = get_transform()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader




