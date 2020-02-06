from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
import argparse

def create_dictionary(dataroot, only_image_questions):
    dictionary = Dictionary()
    questions = []
    files = [
        'official_aaai_split_train_data.json',
        'v2_OpenEnded_mscoco_train2014_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        if path == 'official_aaai_split_train_data.json':
            if only_image_questions:
                qs = [example for example in json.load(open(question_path)) if example['q_type'] == 'image']
            else:
                qs = [example for example in json.load(open(question_path)) if example['image'] is not None]
        else:
            qs = json.load(open(question_path))['questions']
            caps = [dia['caption'] for dia in json.load(open(os.path.join(dataroot, 'visdial_1.0_train.json')))['data']['dialogs']]
            for cap in caps:
                dictionary.tokenize(cap, True)
        for example in qs:
            dictionary.tokenize(example['question'], True)
            if path == 'official_aaai_split_train_data.json':
                dictionary.tokenize(example['image']['caption'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_image_questions', action='store_true', help='Only load image question types')

    args = parser.parse_args()

    d = create_dictionary('data', args.only_image_questions)
    d.dump_to_file('data/dictionary.pkl')

    d = Dictionary.load_from_file('data/dictionary.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('data/glove6b_init_%dd.npy' % emb_dim, weights)
