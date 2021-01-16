import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os

from config import load_config
from preprocess import load_data
from model import Model
import clip


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

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_loader.dataset.classes])
    if args.cuda:
        text_inputs = text_inputs.cuda()

    losses = 0.
    for i, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        img_feats = model.encode_image(data)
        select_text_inputs = text_inputs[target, :]
        text_feats = model.encode_text(select_text_inputs)

        temperature_factor = torch.tensor(args.temperature_factor)
        if args.cuda:
            temperature_factor = temperature_factor.cuda()
        logits = torch.matmul(img_feats, text_feats.T) * torch.exp(temperature_factor)

        labels = torch.arange(target.size(0))

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        losses += loss.item()
        optimizer.step()
        print('[Epoch: {}], losses: {}'.format(epoch, losses / len(train_loader.dataset)))


def main(args):
    model = Model(args.out_channels)

    if args.cuda:
        model = model.cuda()
    args.input_resolution = 32
    train_data, train_loader, test_data, test_loader = load_data(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    for epoch in range(1, args.epochs + 1):
        _train(epoch, model, train_loader, optimizer, criterion, args)
        save_checkpoint(model, optimizer, args, epoch)


if __name__ == '__main__':
    args = load_config()
    main(args)
