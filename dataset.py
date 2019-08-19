from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

MAX_VOCAB_SIZE = 12000


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.add_word('[UNK]')

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w] if w in self.word2idx else self.word2idx['[UNK]'])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            if len(self.word2idx) >= MAX_VOCAB_SIZE:
                return self.word2idx['[UNK]']
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer, name):
    answer.pop('image_id')
    answer.pop('question_id')
    if name == 'val' or name == 'test' or name == 'finetune' or name == 'dev':
        entry = {
            'question_id': question['id'],
            'image_id': question['id'],
            'image': img,
            'caption': question['image']['caption'],
            'question': question['question'],
            'answer': answer}
    else:
        entry = {
            'question_id': question['question_id'],
            'image_id': question['image_id'],
            'image': img,
            'caption': None,
            'question': question['question'],
            'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if name == 'val' or name == 'test' or name == 'dev' or name == 'finetune':
        if name == 'val' or name == 'test':
            question_path = os.path.join(dataroot, 'new_id_format_test_data.json')
            # questions = sorted([ex for ex in json.load(open(question_path))], key=lambda x: x['id'])
            questions = sorted([ex for ex in json.load(open(question_path)) if ex['image'] is not None],
                               key=lambda x: x['id'])
        elif name == 'dev':
            question_path = os.path.join(dataroot, 'new_id_format_dev_data.json')
            # questions = sorted([ex for ex in json.load(open(question_path))], key=lambda x: x['id'])
            questions = sorted([ex for ex in json.load(open(question_path)) if ex['q_type'] == 'image'],
                               key=lambda x: x['id'])
        elif name == 'finetune':
            question_path = os.path.join(dataroot, 'new_id_format_train_data.json')
            questions = sorted([ex for ex in json.load(open(question_path)) if ex['q_type'] == 'image'],
                               key=lambda x: x['id'])
    else:
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
        questions = sorted(json.load(open(question_path))['questions'],
                           key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    num_missing = 0
    for question, answer in zip(questions, answers):
        if name == 'val' or name == 'test' or name == 'finetune' or name == 'dev':
            # utils.assert_eq(question['id'], answer['question_id'])
            if question['id'] != answer['question_id']:
                continue
        else:
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            # if question['question_id'] != answer['question_id'] or question['image_id'] != answer['image_id']:
            #    continue
        img_id = question['id'] if name == 'val' or name == 'test' or name == 'dev' or name == 'finetune' else question[
            'image_id']
        if img_id not in img_id2val:
            num_missing += 1
            continue
        entries.append(_create_entry(img_id2val[img_id], question, answer, name))

    if name == 'val' or name == 'test' or name == 'finetune' or name == 'dev':
        print('Missing ' + str(num_missing) + ' of our examples')


    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, device, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test', 'dev', 'finetune']

        self.name = name
        self.device = device

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        if self.name == 'test' or self.name == 'finetune' or self.name == 'dev':
            self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, 'val36_imgid2idx.pkl'), 'rb'))
        else:
            self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, name + '36_imgid2idx.pkl'), 'rb'))

        print('loading features from h5 file')
        if self.name == 'test' or self.name == 'finetune' or self.name == 'dev':
            h5_path = os.path.join(dataroot, 'val36.hdf5')
        else:
            h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        hf = h5py.File(h5_path, 'r')
        self.features = hf.get('image_features')
        self.spatials = hf.get('spatial_features')

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.shape[2]
        self.s_dim = self.spatials.shape[2]

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

            if entry['caption']:
                tokens = self.dictionary.tokenize(entry['caption'], False)
                tokens = tokens[:50]
                if len(tokens) < 50:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (50 - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), 50)
                entry['c_token'] = tokens
            else:
                entry['c_token'] = [-1] * 50

    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            caption = torch.from_numpy(np.array(entry['c_token']))
            entry['c_token'] = caption

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = torch.from_numpy(self.features[entry['image']])
        spatials = torch.from_numpy(self.spatials[entry['image']])

        question = entry['q_token']
        caption = entry['c_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.name == 'val' or self.name == 'test' or self.name == 'dev':
            if labels is None:
                labels = torch.tensor([-1])
            return entry['question_id'], features, spatials, question, caption, target, labels
        else:
            return features, spatials, question, caption, target

    def __len__(self):
        return len(self.entries)
