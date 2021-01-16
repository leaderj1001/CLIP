import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, out_channels):
        super(Model, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = nn.Sequential(*self.img_model)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc = nn.Linear(self.img_backbone.inplanes, out_channels)

        ntoken, ninp, nhead, nhid, nlayers, dropout = 49408, 768, 8, 2048, 12, 0.5
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.text_fc = nn.Linear(ninp, out_channels)

    def forward(self, img, text):
        n_batch = img.size(0)

        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)

        src = self.encoder(text) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        text_out = self.transformer_encoder(src, None)

        text_out = text_out[:, -1, :]
        text_out = self.text_fc(text_out)
        text_out = F.normalize(text_out, p=2, dim=-1)

        return img_out, text_out

    def encode_image(self, image):
        n_batch = image.size(0)

        out = self.img_model(image)
        out = self.avg_pool(out)
        out = out.view(n_batch, -1)
        out = self.img_fc(out)

        return out

    def encode_text(self, text):
        src = self.encoder(text) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, None)

        out = out[:, -1, :]
        out = self.text_fc(out)

        return out


# x = torch.randn([2, 3, 32, 32])
# t = torch.randint(10, size=[2, 16])
#
# m = Model(512)
# print(m.encode_image(x).size(), m.encode_text(t).size())