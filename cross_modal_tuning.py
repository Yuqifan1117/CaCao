import argparse
import json
from numpy import float32
from tqdm import tqdm

from models.MLM.mpt_test import VisualBertPromptModel
from models.MLM.utils import fineTuningDataset

from models.MLM.tokenization_bert_fast import BertTokenizerFast
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

def train_epoch(model, train_loader, optimizer, lr_scheduler, device):
    total_train_loss = 0
    torch.autograd.set_detect_anomaly(True)
    for triplets in train_loader:
        batch_text = []
        batch_img = []
        for i in range(len(triplets[1])):
            subject, predicate, object = triplets[1][i].split('--')
            batch_text.append((subject.lower(), predicate.lower(), object.lower()))
            batch_img.append(triplets[0][i])      
        outputs, label = model(batch_text, batch_img, weight, device)
        # loss = outputs.loss
        loss, logits = outputs[:2]
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        lr_scheduler.step()
        optimizer.step()
    total_mean_loss = total_train_loss / len(train_loader)
    return total_train_loss, total_mean_loss

def train(model, train_loader, val_loader, config, mode):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=180, num_training_steps=1800, last_epoch=-1)
    # total_epochs = 0
    print('------start training---------')
    param_name = ['transformerlayer', 'prompt4re', 'model.bert.embeddings.word_embeddings']
    # param_name = ['transformerlayer']

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for p in param_name:
            if p in name:
                param.requires_grad = True
                break
        # pytorch_total_params = sum(p.numel() for p in model.parameters())
        # trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('total params: %.2f M, trainable params: %.2f M' % (pytorch_total_params / 1000000.0, trainable_pytorch_total_params / 1000000.0))
    for total_epochs in range(0, 35):
        model.train()
        total_loss, total_mean_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, device=config.device)
        print('total_epochs:{iter} {avg_loss}'.format(iter=total_epochs,avg_loss=total_mean_loss))
        if total_epochs % 1 == 0:
            print('------start evaluation---------')
            model.eval()
            with torch.no_grad():
                recall_1, recall_5, recall_10, val_loss = eval(model, val_loader, device=config.device)
                print('Recall@1:{r1}   Recall@5:{r5}   Recall@10:{r10}'.format(r1=recall_1,r5=recall_5,r10=recall_10))
                # lr_scheduler.step(recall_1)
    
def eval(model, val_loader, device):
    top_10 = 0
    top_5 = 0
    top_1 = 0
    total = 0
    cluster_dict = json.load(open('utils_data/cluster/CaCao_all_cluster_dict_07.json','r'))
    for triplets in tqdm(val_loader):
        batch_text = []
        batch_img = []
        for i in range(len(triplets[1])):
            subject, predicate, object = triplets[1][i].split('--')
            batch_text.append((subject.lower(), predicate.lower(), object.lower()))
            batch_img.append(triplets[0][i])        
        val_output, label = model(batch_text, batch_img, weight, device)
        predictions = val_output[1]
        val_loss = val_output[0]
        for j in range(predictions.shape[0]):
            label_j = label[j]
            word_1 = []
            word_5 = []
            word_10 = []
            # if torch.max(torch.softmax(predictions[j], dim=0)).item() < 0.2:
            #     continue
            word_candidates_1 = torch.argsort(predictions[j], descending=True)[:1].tolist()
            word_candidates_5 = torch.argsort(predictions[j], descending=True)[:5].tolist()
            word_candidates_10 = torch.argsort(predictions[j], descending=True)[:10].tolist()

            for k in word_candidates_1:
                for c in cluster_dict.keys():
                    if words[k] in cluster_dict[c]['words']:
                        for w in cluster_dict[c]['words']:
                            word_1.append(w)
                        break
                # word_1.append(words[k])
            for k in word_candidates_5:
                for c in cluster_dict.keys():
                    if words[k] in cluster_dict[c]['words']:
                        for w in cluster_dict[c]['words']:
                            word_5.append(w)
                        break
                # word_5.append(words[k])
            for k in word_candidates_10:
                for c in cluster_dict.keys():
                    if words[k] in cluster_dict[c]['words']:
                        for w in cluster_dict[c]['words']:
                            word_10.append(w)
                        break
                # word_10.append(words[k])
            for c in cluster_dict.keys():
                if words[label_j.item()] in cluster_dict[c]['words']:
                    label_c = cluster_dict[c]['represent_word']
                    break
            # label_c = words[label_j.item()]
            if label_c in word_1:
                top_1 += 1
            if label_c in word_5:
                top_5 += 1
            if label_c in word_10:
                top_10 += 1
            total += 1
    if total > 0:
        recall_1 = top_1 / total
        recall_5 = top_5 / total
        recall_10 = top_10 / total
    else:
        recall_1 = 0
        recall_5 = 0
        recall_10 = 0
    return recall_1, recall_5, recall_10, val_loss

def test(model, test_loader, device):
    top_1 = 0
    top_3 = 0
    top_5 = 0
    top_10 = 0
    len_test = len(test_loader)
    for triplets in tqdm(test_loader):
        batch_text = []
        batch_img = []
        for i in range(len(triplets[1])):
            subject, predicate, object = triplets[1][i].split('--')
            batch_text.append((subject.lower(), predicate.lower(), object.lower()))
            batch_img.append(triplets[0][i])   
        output, label = model(batch_text, batch_img, weight, device) # output contain loss and hidden representation
        predictions = output[1]
        if torch.max(torch.softmax(predictions[0], dim=0)).item() < 0.1:
            len_test -= 1
            continue
        cluster_dict = json.load(open('utils_data/cluster/CaCao_all_cluster_dict_07.json','r'))
        word_1 = []
        word_3 = []
        word_5 = []
        word_10 = []
        word_candidates_1 = torch.argsort(predictions[0], descending=True)[:1].tolist()
        word_candidates_3 = torch.argsort(predictions[0], descending=True)[:3].tolist()
        word_candidates_5 = torch.argsort(predictions[0], descending=True)[:5].tolist()
        word_candidates_10 = torch.argsort(predictions[0], descending=True)[:10].tolist()
        for k in word_candidates_1:
            for c in cluster_dict.keys():
                if words[k] in cluster_dict[c]['words']:
                    for w in cluster_dict[c]['words']:
                        word_1.append(w)
                    break
        for k in word_candidates_3:
            for c in cluster_dict.keys():
                if words[k] in cluster_dict[c]['words']:
                    for w in cluster_dict[c]['words']:
                        word_3.append(w)
                    break
        for k in word_candidates_5:
            for c in cluster_dict.keys():
                if words[k] in cluster_dict[c]['words']:
                    for w in cluster_dict[c]['words']:
                        word_5.append(w)
                    break
        for k in word_candidates_10:
            for c in cluster_dict.keys():
                if words[k] in cluster_dict[c]['words']:
                    for w in cluster_dict[c]['words']:
                        word_10.append(w)
                    break

        if words[label.item()] in word_1:
            top_1 += 1
        if words[label.item()] in word_3:
            top_3 += 1
        if words[label.item()] in word_5:
            top_5 += 1
        if words[label.item()] in word_10:
            top_10 += 1
    print('top_1 acc: ', top_1 / len_test)
    print('top_3 acc: ', top_3 / len_test)
    print('top_5 acc: ', top_5 / len_test)
    print('top_10 acc: ', top_10 / len_test)
    return top_1 / len_test
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-modal Predicate Boosting")
    parser.add_argument(
        "--lr",
        type=float32,
        default=1e-2,
    )
    parser.add_argument(
        "--weight_decay",
        type=float32,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=float32,
        default=5,
    )
    parser.add_argument(
        "--factor",
        type=float32,
        default=0.1,
    )
    parser.add_argument(
        "--device",
        default="cuda:2"
    )
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained('/home/qifan/FG-SGG_from_LM/bert-base-uncased')
    prompt_candidates = []
    with open('bert-base-uncased/prompt.txt','r') as f:
        for line in f.readlines():
            prompt_candidates.append(line.strip('\n'))
    prompt_num = 10

    # gqa preparation

    train_dataset = fineTuningDataset('datasets/image_caption_triplet_all.json',"/home/qifan/datasets/coco/train2014/",'train')
    weight = train_dataset.weight
    words = train_dataset.predicates_words
    relation_type_count = len(words)
    print(relation_type_count)
    val_dataset = fineTuningDataset('datasets/image_caption_triplet_all.json',"/home/qifan/datasets/coco/train2014/",'val')
    test_dataset = fineTuningDataset('datasets/image_caption_triplet_all.json',"/home/qifan/datasets/coco/train2014/",'test')
    # vg_words = [line.strip('\n').strip('\r') for line in open('/home/qifan/datasets/vg/predicate_list.txt')]
    print('train:{train}, val:{val}, test:{test}'.format(train=len(train_dataset),val=len(val_dataset),test=len(test_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)
    model_own = VisualBertPromptModel(prompt_num, prompt_candidates, words, relation_type_count=relation_type_count)
    # train
    train(model_own, train_loader, val_loader, args, 'VPT')
    torch.save(model_own.state_dict(),'pre_trained_visually_prompted_model/ablation_labelsmoothing/cluster_50_model_VPT_threshold09_test_3.pkl') 
    recall_1, recall_5, recall_10, val_loss= eval(model_own, test_loader, device=args.device)
    print('Recall@1:{r1}   Recall@5:{r5}   Recall@10:{r10}  val_loss:{loss}'.format(r1=recall_1,r5=recall_5,r10=recall_10,loss=val_loss))   
    # test
    model_own.load_state_dict(torch.load('pre_trained_visually_prompted_model/ablation_labelsmoothing/cluster_50_model_VPT_threshold09_test_3.pkl'), strict=False)
    print('test recall@1:{}'.format(test(model_own, test_loader, args.device)))
    
