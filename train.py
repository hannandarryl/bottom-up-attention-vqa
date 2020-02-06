import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import pickle
import json


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, lr, device):
    model.train()
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters(), lr=lr)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, c, a) in tqdm(enumerate(train_loader)):
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            c = c.to(device)
            a = a.to(device)

            pred = model(v, q, c)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data, device).sum()
            total_loss += loss.data.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.eval()
        eval_score, bound = evaluate(model, eval_loader, device, 'dev')
        model.train()

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader, device, set_name):
    model.eval()
    score = 0
    upper_bound = 0
    num_data = 0
    lbl_to_ans = pickle.load(open('data/cache/trainval_label2ans.pkl', 'rb'))
    if set_name == 'test':
        id_to_a = {ex['id']: ex['answer'] for ex in json.load(open('data/official_aaai_split_test_img_examples.json'))}
    elif set_name == 'dev':
        id_to_a = {ex['id']: ex['answer'] for ex in json.load(open('data/official_aaai_split_dev_data.json')) if ex['q_type'] == 'image'}
    if set_name == 'test':
        out_file = open('official_aaai_split_test_modality_selector_image_predictions.txt', 'w+')
    elif set_name == 'dev':
        out_file = open('tmp.txt', 'w+')
    with torch.no_grad():
        for id_list, v, b, q, c, a, label in iter(dataloader):
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            c = c.to(device)
            #pickle.dump(v, open('v.pkl', 'wb+'))
            #pickle.dump(q, open('q.pkl', 'wb+'))
            #pickle.dump(c, open('c.pkl', 'wb+'))
            #print(torch.sum(pickle.load(open('v.pkl', 'rb')) - v))
            #print(torch.sum(pickle.load(open('q.pkl', 'rb')) - q))
            #print(torch.sum(pickle.load(open('c.pkl', 'rb')) - c))
            #raise Exception

            pred = model(v, q, c)
            # batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            # score += batch_score
            for p, l, id_num in zip(pred, label, id_list):
                guess = torch.argmax(p).item()
                out_file.write(id_num + '\t' + lbl_to_ans[guess] + '\n')
                if lbl_to_ans[guess].lower().strip() == id_to_a[id_num].lower().strip():
                    score += 1
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
    
    #for k, v in model.state_dict().items():
    #    old_v = pickle.load(open(k+'.pkl', 'rb'))
    #    print(torch.sum(v - old_v))
    
    print('Number correct: ' + str(score))
    score = score / len(id_to_a)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
