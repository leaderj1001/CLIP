import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=77)
    parser.add_argument('--checkpoints', type=str, default=None)

    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--device_num', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--out_channels', type=int, default=512)
    parser.add_argument('--temperature_factor', type=float, default=0.07)

    args = parser.parse_args()

    return args
