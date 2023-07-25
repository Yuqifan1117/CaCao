import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


def train_epoch(model, train_loader, optimizer, theta, device):
    total_train_loss = 0
    torch.autograd.set_detect_anomaly(True)
    for triplets in tqdm(train_loader):
        batch_text = []
        batch_img = []
        for i in range(len(triplets[1])):
            subject, predicate, object = triplets[1][i].split('--')
            batch_text.append((subject.lower(), predicate.lower(), object.lower()))
            batch_img.append(triplets[0][i])      
        outputs, label = model(batch_text, batch_img, weight, theta, device)
        # loss = outputs.loss
        loss, logits = outputs[:2]
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_mean_loss = total_train_loss / len(train_loader)
    return total_mean_loss

def train(model, train_loader, val_loader, config, mode=None):
    for name, param in model.named_parameters():
        if mode == 'VPT':
            param_name = ['transformerlayer']
        elif mode == 'ASCL':
            param_name = ['prompt4re']
        elif mode == 'LPT': 
            param_name = ['model.bert.embeddings.word_embeddings']
        else:
            param_name = ['transformerlayer', 'prompt4re', 'model.bert.embeddings.word_embeddings']
        for p_name in param_name:
            if p_name in name:
                param.requires_grad = True
                break
            else:
                param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=4230, num_training_steps=42300, last_epoch=-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.patience, factor=config.factor
    )
    # total_epochs = 0
    print('------start training---------')
    for total_epochs in range(0, 5):
        model.train()
        total_mean_loss = train_epoch(model, train_loader, optimizer, config.theta, device=config.device)
        print('total_epochs:{iter} {avg_loss}'.format(iter=total_epochs,avg_loss=total_mean_loss))
        if total_epochs % 1 == 0:
            print('------start evaluation---------')
            model.eval()
            with torch.no_grad():
                recall_1, recall_10, val_loss = eval(model, val_loader, config.theta, device=config.device)
                print('val_Recall@1:{r1} val_Recall@10:{r10}'.format(r1=recall_1,r10=recall_10))
                # l
                lr_scheduler.step(recall_1)
    
def eval(model, val_loader, theta, device):
    top_10 = 0
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
        val_output, label = model(batch_text, batch_img, weight, theta, device)
        predictions = val_output[1]
        val_loss = val_output[0]
        for j in range(predictions.shape[0]):
            label_j = label[j]
            word_1 = []
            word_10 = []
            word_candidates_1 = torch.argsort(predictions[j], descending=True)[:1].tolist()
            word_candidates_10 = torch.argsort(predictions[j], descending=True)[:10].tolist()

            for k in word_candidates_1:
                for c in cluster_dict.keys():
                    if words[k] in cluster_dict[c]['words']:
                        for w in cluster_dict[c]['words']:
                            word_1.append(w)
                        break
                # word_1.append(words[k])
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
            if label_c in word_10:
                top_10 += 1
            total += 1
    if total > 0:
        recall_1 = top_1 / total
        recall_10 = top_10 / total
    else:
        recall_1 = 0
        recall_10 = 0
    return recall_1, recall_10, val_loss

def test(model, test_loader, theta, device):
    top_1 = 0
    top_10 = 0
    len_test = len(test_loader)
    for triplets in tqdm(test_loader):
        batch_text = []
        batch_img = []
        for i in range(len(triplets[1])):
            subject, predicate, object = triplets[1][i].split('--')
            batch_text.append((subject.lower(), predicate.lower(), object.lower()))
            batch_img.append(triplets[0][i])   
        output, label = model(batch_text, batch_img, weight, theta, device) 
        predictions = output[1]
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
        for k in word_candidates_10:
            for c in cluster_dict.keys():
                if words[k] in cluster_dict[c]['words']:
                    for w in cluster_dict[c]['words']:
                        word_10.append(w)
                    break
        # for k in word_candidates_1:
        #     word_1.append(words[k])
        # for k in word_candidates_3:
        #     word_3.append(words[k])
        # for k in word_candidates_5:
        #     word_5.append(words[k])

        if words[label.item()] in word_1:
            top_1 += 1
        if words[label.item()] in word_10:
            top_10 += 1
    print('top_1 acc: ', top_1 / len_test)
    print('top_10 acc: ', top_10 / len_test)
    return top_1 / len_test

def main(args):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)
    model_own = VisualBertPromptModel(args.prompt_num, prompt_candidates, words, relation_type_count=relation_type_count)
    # train
    train(model_own, train_loader, val_loader, args)
    torch.save(model_own.state_dict(),'checkpoints/cluster_50_model.pkl') 
    recall_1, recall_5, recall_10, val_loss= eval(model_own, test_loader, device=args.device)
    print('Recall@1:{r1}   Recall@5:{r5}   Recall@10:{r10}  val_loss:{loss}'.format(r1=recall_1,r5=recall_5,r10=recall_10,loss=val_loss))   

    test_acc = test(model_own, test_loader, args.theta, args.device)
    print('test recall@1:{}'.format(test_acc))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-modal Predicate Boosting")
    parser.add_argument(
        "--lr",
        type=float32,
        default=1.5e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float32,
        default=7.5e-4,
    )
    parser.add_argument(
        "--patience",
        type=float32,
        default=2,
    )
    parser.add_argument(
        "--factor",
        type=float32,
        default=0.5,
    )
    parser.add_argument(
        "--device",
        default="cuda:0"
    )
    parser.add_argument(
        "--batch_size",
        default=64
    )
    parser.add_argument(
        "--theta",
        default=8.0
    )
    parser.add_argument(
        "--prompt_num",
        default=10
    )
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    prompt_candidates = []
    with open('bert-base-uncased/prompt.txt','r') as f:
        for line in f.readlines():
            prompt_candidates.append(line.strip('\n'))

    train_dataset = fineTuningDataset('datasets/image_caption_triplet.json',"/home/qifan/datasets/coco/train2014/",'train')
    val_dataset = fineTuningDataset('datasets/image_caption_triplet.json',"/home/qifan/datasets/coco/train2014/",'val')
    test_dataset = fineTuningDataset('datasets/image_caption_triplet.json',"/home/qifan/datasets/coco/train2014/",'test')
    print('train:{train}, val:{val}, test:{test}'.format(train=len(train_dataset),val=len(val_dataset),test=len(test_dataset)))
    weight = train_dataset.weight
    words = train_dataset.predicates_words
    relation_type_count = len(words)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    # train
    main(args)
    # test recall@1:0.7357060518731989
    
