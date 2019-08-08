import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
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
    #torch.backends.cudnn.benchmark = True

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:' + args.gpu)

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, device)
    eval_dset = VQAFeatureDataset('test', dictionary, device)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = model.to(device)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=6)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=6)
    if not args.model_ckpt:
        train(model, train_loader, eval_loader, args.epochs, args.output, device)
    else:
        model.load_state_dict(torch.load(args.model_ckpt))
        eval_score, bound = evaluate(model, eval_loader)
        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
