import json
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel
import numpy as np
from transformers import BertModel, BertTokenizerFast
from models.MLM.utils import layer_init

class BertForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config, relation_type_count, mlm_head, words, predicate_embeddings):
        super().__init__(config)
        self.cls = mlm_head
        self.word_table = words
        self.classifier = nn.Sequential(
            nn.LeakyReLU(1e-2),
            nn.Linear(config.vocab_size, relation_type_count),
            nn.Dropout(0.1)
        )
        self.predicate_embeddings = predicate_embeddings
        layer_init(self.classifier[1], xavier=True)
    def forward(
        self,
        h_cls,
        mask_pos=None,
        labels=None,
        weight=None,
        theta=None,
        device='cuda:0'
    ):

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        sequence_mask_output = h_cls[torch.arange(h_cls.size(0)), mask_pos] # only predict the masked position
        sequence_mask_scores = self.cls(sequence_mask_output)
        # sequence_mask_scores = sequence_mask_output
        logits = self.classifier(sequence_mask_scores)
        loss = None
        # adaptive semantic cluster loss
        if labels is not None:
            cluster_label = F.one_hot(labels.view(-1), num_classes=logits.shape[-1]).float()
            # semantic clustering
            cluster_dict = json.load(open('utils_data/cluster/CaCao_all_cluster_dict_07.json','r'))
            for j in range(labels.shape[0]):
                label_w = self.word_table[labels[j].item()]
                target_predicate_embedding = self.predicate_embeddings[labels[j].item()].to(device) # target predicate
                for c in cluster_dict.keys():
                    if label_w in cluster_dict[c]['words']:
                        c_num = len(cluster_dict[c]['words'])
                        if c_num > 1:
                            for w in cluster_dict[c]['words']:
                                if w != label_w:
                                    k = self.word_table.index(w)
                                    cluster_predicate_embedding = self.predicate_embeddings[k].to(device)
                                    s_ij = F.cosine_similarity(target_predicate_embedding, cluster_predicate_embedding)
                                    cluster_label[j][k] = s_ij*1/c_num 
                        break
            loss = own_ce(logits.view(-1, logits.size(-1)), cluster_label, weight, theta)
        output = (logits,)
        return ((loss,) + output) if loss is not None else output
def own_ce(x, soft_cluster, weight, theta):
    if weight is None:
        LogSoftmax = F.log_softmax(x, 1)
    else:
        # process for weight
        total_weight = []
        for i in range(soft_cluster.shape[0]):
            k = torch.argmax(soft_cluster, dim=1)[i].item()
            total_weight.append(weight*1/weight[k])
        total_weight = torch.stack(total_weight, dim=0)
        e_x = torch.exp(x - torch.max(x, dim=-1)[0].unsqueeze(1).repeat(1, x.shape[1]))
        weighted_logits = e_x  * adaptive_weight(x, soft_cluster, total_weight, theta)
        # LogSoftmax = torch.log(e_x / e_x.sum(dim=-1).reshape(-1,1)+1e-8)
        LogSoftmax = torch.log((e_x /torch.sum(weighted_logits, dim=1).reshape(-1,1))+1e-8)
    result = soft_cluster*LogSoftmax
    nllloss = -torch.mean(result, dim=1)
    nllloss = torch.mean(nllloss, dim=0)
    return nllloss
def adaptive_weight(x, label, weight, theta):
    x = tensor_normalize(x)+1.0
    i_idx = []
    for i in range(label.shape[0]):
        k = torch.argmax(label, dim=1)[i].item()
        # k = torch.where(label[i]==0.9)[0].item()
        i_idx.append([k])
    z_i = torch.gather(x, dim=1, index=torch.tensor(i_idx, device=x.device).long())
    z_i_rec = torch.reciprocal(z_i)
    z_j = x.clone()
    temp = torch.mul(z_j, z_i_rec) * theta
    weight_logits = temp * weight
    return weight_logits

class TripletLoss_CB(nn.Module):

    def __init__(self, margin):
        super(TripletLoss_CB, self).__init__()
        self.margin = margin

    def forward(self, pos_samp, neg_samp, weights):
        weights_sum = weights.sum()
        losses = ((F.relu(neg_samp - pos_samp + self.margin) * weights).mean(dim=1)).sum() / weights_sum
        return losses

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx_output = r_mat_inv.dot(mx)
    return mx_output
def tensor_normalize(data):
    d_min = data.min(dim=1)[0]
    data += torch.abs(d_min).unsqueeze(1).repeat(1,data.shape[1])
    d_min = data.min(dim=1)[0]
    d_max = data.max(dim=1)[0]
    dst = d_max - d_min
    norm_data = (data - d_min.unsqueeze(1).repeat(1,data.shape[1])).true_divide(dst.unsqueeze(1).repeat(1,data.shape[1]))
    return norm_data
def adj_laplacian(adj):
    adj = normalize(adj)
    return adj