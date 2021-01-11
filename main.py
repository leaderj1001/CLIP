import torch
import torch.nn.functional as F

import numpy as np
import urllib.request
import os

from config import load_config
from preprocess import load_data
from simple_tokenizer import SimpleTokenizer

MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

DATA = {
    "bpe_simple_vocab_16e6": "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
}


def _download():
    print('Model Download...')
    if not os.path.isdir('model'):
        os.mkdir('model')
    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isfile('./model/model.pt'):
        urllib.request.urlretrieve(MODELS['ViT-B/32'], './model/model.pt')
    if not os.path.isfile('./data/bpe_simple_vocab_16e6.txt.gz'):
        urllib.request.urlretrieve(DATA['bpe_simple_vocab_16e6'], './data/bpe_simple_vocab_16e6.txt.gz')


def _eval(model, text_features, test_loader, args):
    model.eval()

    total_correct, step = 0., 0.
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            img_features = model.encode_image(data).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity = 100. * (img_features @ text_features.T)
            probs = F.softmax(similarity, dim=-1).max(-1)[1]

            total_correct += probs.eq(target).sum().item()
            step += target.size(0)
    print('Step: {}, Acc: {}'.format(step, total_correct / step * 100.))


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
    _download() # model, bpe data download
    model = torch.jit.load("./model/model.pt", map_location='cpu') # pytorch 1.7.1
    for name, _ in model.named_parameters():
        print(name)
    if args.cuda:
        model = model.cuda()
    input_resolution = model.input_resolution.item()
    args.input_resolution = 224
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    test_data, test_loader = load_data(args)

    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    text_descriptions = [f"This is a photo of a {label}" for label in test_data.classes]
    print(text_descriptions)
    text_tokens = [[sot_token] + tokenizer.encode(desc) + [eot_token] for desc in text_descriptions]
    text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)

    for i, tokens in enumerate(text_tokens):
        text_input[i, :len(tokens)] = torch.tensor(tokens)

    if args.cuda:
        text_input = text_input.cuda()

    text_features = model.encode_text(text_input).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    _eval(model, text_features, test_loader, args)


if __name__ == '__main__':
    args = load_config()
    main(args)
