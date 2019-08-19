import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--finetune_epochs', type=int, default=2)
    parser.add_argument('--num_hid', type=int, default=2048)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--finetune_lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--model_ckpt', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:' + args.gpu)

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, device)
    finetune_dset = VQAFeatureDataset('finetune', dictionary, device)
    dev_dset = VQAFeatureDataset('dev', dictionary, device)
    eval_dset = VQAFeatureDataset('test', dictionary, device)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = model.to(device)

    if args.model_ckpt:
        model.load_state_dict(torch.load(args.model_ckpt))

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=6)
    finetune_loader = DataLoader(finetune_dset, batch_size, shuffle=True, num_workers=6)
    dev_loader = DataLoader(dev_dset, batch_size, shuffle=False, num_workers=6)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=6)
    if args.train:
        train(model, train_loader, dev_loader, args.epochs, args.output, args.lr, device)
        model.load_state_dict(torch.load(os.path.join(args.output, 'model.pth')))
    if args.finetune:
        train(model, finetune_loader, dev_loader, args.finetune_epochs, os.path.join(args.output, 'finetune'),
              args.finetune_lr, device)

    eval_score, bound = evaluate(model, eval_loader, device, 'test')
    print('\tTest score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
