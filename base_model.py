import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, c_emb, v_att, q_net, v_net, c_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.v_att = v_att
        #self.v_q_att = v_q_att
        #self.v_c_att = v_c_att
        self.q_net = q_net
        self.v_net = v_net
        #self.v_q_net = v_q_net
        #self.v_c_net = v_c_net
        self.c_net = c_net
        self.classifier = classifier

    def forward(self, v, q, c):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        c_w = self.w_emb(c)
        c_emb = self.c_emb(c_w)

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        c_repr = self.c_net(c_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = torch.cat([q_repr * v_repr, c_emb], dim=1)
        logits = self.classifier(joint_repr)
        #logits = self.classifier(q_repr)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    c_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    c_net = FCNet([num_hid, num_hid])
    classifier = SimpleClassifier(
        2* num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, c_emb, v_att, q_net, v_net, c_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    c_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    c_net = FCNet([c_emb.num_hid, num_hid])
    classifier = SimpleClassifier(
        2*num_hid, 2*num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, c_emb, v_att, q_net, v_net, c_net, classifier)
