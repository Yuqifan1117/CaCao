import json
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel
import numpy as np
from transformers import BertModel, BertTokenizerFast
from models.MLM.utils import layer_init

class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config, relation_type_count, mlm_head, words):
        super().__init__(config)
        self.cls = mlm_head
        self.word_table = words
        self.classifier = nn.Sequential(
            nn.LeakyReLU(1e-2),
            nn.Linear(config.vocab_size, relation_type_count),
            nn.Dropout(0.1)
        )
        self.model = BertModel.from_pretrained('/home/qifan/FG-SGG_from_LM/bert-base-uncased')
        self.embedding = BertModel.from_pretrained('/home/qifan/FG-SGG_from_LM/bert-base-uncased').get_input_embeddings()
        self.tokenizer = BertTokenizerFast.from_pretrained('/home/qifan/FG-SGG_from_LM/bert-base-uncased')
        layer_init(self.classifier[1], xavier=True)
    def forward(
        self,
        h_cls,
        mask_pos=None,
        labels=None,
        contexts=None,
        weight=None,
        device='cuda:0'
    ):

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        sequence_mask_output = h_cls[torch.arange(h_cls.size(0)), mask_pos] # only predict the masked position
        sequence_mask_scores = self.cls(sequence_mask_output)
        # sequence_mask_scores = sequence_mask_output
        logits = self.classifier(sequence_mask_scores)
        loss = None
        # label smoothing with adaptive weight
        # if labels is not None:
        #     onehot_label = F.one_hot(labels.view(-1), num_classes=logits.shape[-1]).float()
        #     smoothing = 0.1
        #     smooth_label = (1-smoothing) * onehot_label + smoothing/(onehot_label.shape[-1]-1) * (torch.ones_like(onehot_label)-onehot_label)
        #     weight = None # only test for label smoothing
        #     # log_logits = torch.log(torch.exp(logits.view(-1, logits.size(-1)))/torch.sum(torch.exp(logits.view(-1, logits.size(-1))), dim=1).reshape(-1,1))
        #     # loss = smooth_label*log_logits
        #     # loss = -torch.mean(smooth_label*log_logits, dim=-1)
        #     # loss = torch.mean(loss)
        #     loss = own_ce(logits.view(-1, logits.size(-1)), smooth_label, weight)
        # adaptive semantic cluster loss
        if labels is not None:
            cluster_label = F.one_hot(labels.view(-1), num_classes=logits.shape[-1]).float()
            
            cluster_label = cluster_label #* s_i_so
            cluster_float = 0.1
            cluster_label = (1-cluster_float) * cluster_label + cluster_float/(cluster_label.shape[-1]-1) * (torch.ones_like(cluster_label)-cluster_label)
            # semantic clustering
            cluster_dict = json.load(open('utils_data/cluster/CaCao_50_cluster_dict_0.json','r'))
            for j in range(labels.shape[0]):
                label_w = self.word_table[labels[j].item()]
                label_j = cluster_label[j][labels[j].item()].item()
                target_predicate_embedding = torch.mean(self.embedding(self.tokenizer.encode(label_w, return_tensors="pt", add_special_tokens = False).to(device)), dim=1) # target predicate
                context_embedding = self.model(**self.tokenizer(contexts[j], return_tensors="pt").to(device))[1]
                predicate_embedding = self.model(**self.tokenizer(label_w, return_tensors="pt").to(device))[1]
                s_i_so = F.cosine_similarity(predicate_embedding, context_embedding)
                cluster_label[j][labels[j].item()] = label_j*s_i_so
                for c in cluster_dict.keys():
                    if label_w in cluster_dict[c]['words']:
                        c_num = len(cluster_dict[c]['words'])
                        if c_num > 1:
                            for w in cluster_dict[c]['words']:
                                if w != label_w:
                                    k = self.word_table.index(w)
                                    cluster_predicate_embedding = torch.mean(self.embedding(self.tokenizer.encode(w, return_tensors="pt", add_special_tokens = False).to(device)), dim=1)
                                    s_ij = F.cosine_similarity(target_predicate_embedding, cluster_predicate_embedding)
                                    cluster_label[j][k] = (1-cluster_float)* s_ij/c_num 
                        break
            loss = own_ce(logits.view(-1, logits.size(-1)), cluster_label, weight)
        output = (logits,)
        return ((loss,) + output) if loss is not None else output
def own_ce(x, soft_cluster, weight):
    # hard encode labels by one-hot
    # LogSoftmax = torch.log(torch.exp(x)/torch.sum(torch.exp(x), dim=1).reshape(-1,1))    
    # if weight is None:
    #     result = soft_cluster*LogSoftmax
    # else:
    #     result = soft_cluster*LogSoftmax*weight

    if weight is None:
        LogSoftmax = torch.log(torch.exp(x)/torch.sum(torch.exp(x), dim=1).reshape(-1,1))
    else:
        # adaptive_w = torch.mean(adaptive_weight(x, soft_cluster, weight), dim=0)
        weighted_logits = torch.exp(x) * adaptive_weight(x, soft_cluster, weight)
        epsilon = 10 ** -44
        predicted_p = torch.exp(x)/torch.sum(weighted_logits, dim=-1).reshape(-1,1)
        predicted_p = predicted_p.sigmoid().clamp(epsilon, 1 - epsilon)

        e_x = torch.exp(x - torch.max(x, dim=-1)[0].unsqueeze(1).repeat(1, x.shape[1]))
        weighted_logits = e_x * adaptive_weight(x, soft_cluster, weight)
        LogSoftmax = torch.log(e_x / e_x.sum(dim=-1).reshape(-1,1)+1e-8)
        
        # LogSoftmax = torch.log(torch.exp(x)/torch.sum(weighted_logits, dim=-1).reshape(-1,1)+1e-8)
    result = soft_cluster*LogSoftmax
    sub_pt = 1 - result
    alpha = 2
    gamma = 3 
    focal_result = -alpha*sub_pt**gamma*LogSoftmax
    nllloss = torch.mean(focal_result, dim=1)
    nllloss = torch.mean(nllloss, dim=0)
    return nllloss
def adaptive_weight(x, label, weight):
    # norm_x = x.clone()
    norm_x = tensor_normalize(x) + 0.01
    # norm_x = F.normalize(x, p=2, dim=1)
    z_j = torch.reciprocal(norm_x)
    z_k = []
    for i in range(label.shape[0]):
        k = torch.argmax(label, dim=1)[i].item()
        z_k.append(norm_x[i][k])
    z_k = torch.Tensor(z_k).reshape(norm_x.shape[0], 1).repeat(1, norm_x.shape[1]).to(z_j.device)
    temp = torch.mul(z_k, z_j).pow(1.0)
    weight_logits = torch.min(torch.ones_like(temp), temp+0.01) * weight
    # weight_logits = (-F.relu(-(temp+0.5)+1)+1) * weight
    # weight_logits = tensor_normalize(weight_logits)
    return weight_logits
# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in 
#         Focal Loss for Dense Object Detection.

#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

#         The losses are averaged across observations for each minibatch.
#     """
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = torch.ones(class_num, 1)
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = alpha
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average

#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)

#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]

#         probs = (P*class_mask).sum(1).view(-1,1)

#         log_p = probs.log()
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 


#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss
class AdaptiveSemanticClusterLoss(object):
    def __init__(self, relation_predictor, candidate_number, margin):
        self.relation_predictor = relation_predictor
        if self.relation_predictor == 'TransformerPredictor':
            # freq-bias has established context-predicate associations, and is used to generate biased predictions of baslines.
            # As we extract baselines' confusion matrix from their biased prediction trained with freq bias, it both consider textual & visual context information for each sample of objects and subjects
            self.pred_adj_np = np.load('/home/qifan/FGPL/misc/conf_mat_transformer_train.npy')
            self.pred_adj_np[0, :] = 0.0
            self.pred_adj_np[:, 0] = 0.0
            self.pred_adj_np[0, 0] = 1.0 # set 1 for ``background'' predicate
            self.pred_adj_np = self.pred_adj_np / (self.pred_adj_np.sum(0)[:, None] + 1e-8)
            self.pred_adj_np = adj_laplacian(self.pred_adj_np) # normalize to [0,1] to get predicate-predicate association
            self.pred_adj_np = torch.from_numpy(self.pred_adj_np).float().cuda()
            self.pred_adj_np_diag = torch.diag(self.pred_adj_np)
        self.candidate_number = candidate_number
        self.margin = margin
        self.criterion_loss_contra_cb = TripletLoss_CB(self.use_contra_distance_loss_value)
    def forward(self, relation_logits, rel_labels):
        relation_logits = F.log_softmax(relation_logits,dim=-1)
        pred_adj_np_index = torch.topk(self.pred_adj_np, self.candidate_number).indices# gather probabilities of hard-to-distinguish predicates in each sample based on predicate correlation
        negtive_index = pred_adj_np_index[rel_labels]
        negative = torch.gather(relation_logits,1,negtive_index.long())
        positive = torch.gather(relation_logits, 1, rel_labels.reshape(rel_labels.size(0),1).long()).repeat(1,self.candidate_number)
        # Balancing Factor
        predicate_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
                                14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0,
                                27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0,
                                40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0}
        rel_label_list = rel_labels.cpu().numpy().tolist()
        rel_label_list.sort()
        for key in rel_label_list:
            predicate_count[key] = predicate_count.get(key, 0) + 1
        dict_value_list = list(predicate_count.values())
        n_j = torch.reciprocal(torch.Tensor(dict_value_list).reshape(51, 1).repeat(1, 51).cuda())
        n_i = torch.Tensor(dict_value_list).reshape(51, 1).repeat(1, 51).cuda().t()
        category_statistics = torch.mul(n_i, n_j).pow(1.0)
        negative_weights = torch.gather(category_statistics[rel_labels], 1, negtive_index.long())
        weights = negative_weights
        # adaptive reweighting factor
        loss_relation_contrastive = self.criterion_loss_contra_cb(positive, negative, weights)
        return loss_relation_contrastive

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