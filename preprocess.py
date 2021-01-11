import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image


def load_data(args):
    if args.dataset == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'CIFAR100':
        test_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_data, test_loader
