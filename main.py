import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import math

from config import load_config
from preprocess import load_data
from model import Model
import urllib.request


def save_checkpoint(model, optimizer, args, epoch):
    print('Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))


def _train(epoch, model, train_loader, optimizer, criterion, args):
    model.train()

    losses, step = 0., 0.
    for i, (img, text) in enumerate(train_loader):
        if args.cuda:
            img, text = img.cuda(), text.cuda()

        img_feats, text_feats = model(img, text)

        logits = torch.matmul(img_feats, text_feats.T) * math.exp(args.temperature_factor)

        labels = torch.arange(text.size(0))

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        losses += loss.item()
        step += 1
        optimizer.step()
        print('[Epoch: {}], losses: {}'.format(epoch, losses / step))


def main(args):
    if not os.path.isdir('data'):
        os.mkdir('data')
    urllib.request.urlretrieve('https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz',
                               filename='./data/bpe_simple_vocab_16e6.txt.gz')

    model = Model(args.out_channels)

    if args.cuda:
        model = model.cuda()
    args.input_resolution = 32
    train_data, train_loader, test_data, test_loader = load_data(args)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    for epoch in range(1, args.epochs + 1):
        _train(epoch, model, train_loader, optimizer, criterion, args)
        save_checkpoint(model, optimizer, args, epoch)


if __name__ == '__main__':
    args = load_config()
    main(args)
