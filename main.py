import torch
import torch.nn.functional as F

import numpy as np
import os

from config import load_config
from preprocess import load_data


def _train(model, train_loader, optimizer, criterion, args):
    model.train()

    for i, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        img = data['img']
        text = data['text']

        img_feats = model.encode_image(img)
        text_feats = model.encode_text(text)

        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

        logits = torch.matmul(img_feats, text_feats.T) * torch.exp(args.temperature_factor)

        labels = torch.arange(target.size(0))

        optimizer.zero_grad()
        loss_i = criterion(logits, labels, dim=0)
        loss_t = criterion(logits, labels, dim=1)

        loss = (loss_i + loss_t) / 2.
        loss.backward()
        optimizer.step()


def main(args):
    pass


if __name__ == '__main__':
    args = load_config()
    main(args)
