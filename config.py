import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--device_num', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--out_channels', type=int, default=512)
    parser.add_argument('--temperature_factor', type=float, default=0.07)

    args = parser.parse_args()

    return args
