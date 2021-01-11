import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--cuda', type=bool, default=False)

    args = parser.parse_args()

    return args
