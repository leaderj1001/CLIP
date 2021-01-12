import torch
import torch.nn.functional as F

import clip
from config import load_config
from preprocess import load_data


def _eval(model, text_inputs, test_loader, args):
    model.eval()

    total_correct, step = 0., 0.
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            text_features = model.encode_text(text_inputs).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            img_features = model.encode_image(data).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity = 100. * (img_features @ text_features.T)
            probs = F.softmax(similarity, dim=-1).max(-1)[1]

            total_correct += probs.eq(target).sum().item()
            step += target.size(0)
            print('Step: {}, Acc: {}'.format(step, total_correct / step * 100.))


# reference
# https://github.com/openai/CLIP
def main(args):
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    model, preprocess = clip.load('ViT-B/32', device)
    args.input_resolution = 224

    test_data, test_loader = load_data(args)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test_data.classes]).to(device)
    if args.cuda:
        text_inputs = text_inputs.cuda()

    _eval(model, text_inputs, test_loader, args)


if __name__ == '__main__':
    args = load_config()
    main(args)
