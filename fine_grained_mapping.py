import json
import torch
import os
from models.MLM.mpt_test import VisualBertPromptModel

from models.MLM.utils import fineTuningDataset

def predict_rel_prompt(model, img_path, subject, object, device):
    predicate = '[MASK]'
    batch_text = []
    batch_img = []
    img_id = 1
    basename =  str(img_id) + '.jpg'
    img_path = os.path.join("/home/qifan/datasets/vg/VG_100K", basename)
    batch_text.append((subject.lower(), predicate, object.lower()))  
    batch_img.append(img_path)
    output = model(batch_text, batch_img, device=device, is_label=False)
    text_prompt = subject + '[MASK]' + object
    predictions = output[0][0]
    expand_results = [] 
    # how to reweight for predict
    word_candidates = torch.argsort(predictions[0], descending=True)[:1].tolist()
    for k in word_candidates:
        # print(batch_text[0][0], model.word_tabel[k], batch_text[0][2])
        expand_results.append((batch_text[0][0], model.word_tabel[k], batch_text[0][2]))    
    return expand_results

# def predicate_embedding(subject, predicate, object):

if __name__ == '__main__':
    target_words = [line.strip('\n').strip('\r') for line in open('/home/qifan/datasets/vg/predicate_list.txt')] # base 50 categories predicates
    coarse_predicates = ['above', 'across', 'against', 'along', 'and', 'at', 'behind', 'between', 'for', 'from', 'has', 'in', 'in front of', 'near', 'of', 'on', 'over', 'to', 'under', 'with']
    fine_predicates = ['attached to', 'belonging to', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'growing on', 'hanging from', 'holding', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'on back of', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears']
    
    # resourced dataset for CaCao cross-modal prompt tuning
    finetuned_dataset = fineTuningDataset('datasets/image_caption_triplet_all.json',"/home/qifan/datasets/coco/train2014/",'train')
    raw_predicates = finetuned_dataset.predicates_words
    # got mapping dict with embedding similarity of BERT Embedding Layer(fine-tuned)
    device = 'cuda:0'
    prompt_candidates = []
    with open('bert-base-uncased/prompt.txt','r') as f:
        for line in f.readlines():
            prompt_candidates.append(line.strip('\n'))
    prompt_num = 10
    model_own = VisualBertPromptModel(prompt_num, prompt_candidates, raw_predicates, relation_type_count=len(raw_predicates)).to(device)
    model_own.load_state_dict(torch.load('pre_trained_visually_prompted_model/gqa_fine/gqa_model_VPT_LPT_ASCL_threshold07.pkl', map_location='cuda:0'))

    # fine-grained mapping
    mapping_dict = dict()

    # GQA-200 mapping
    gqa_200_ID_info = '/home/qifan/datasets/GQA/GQA_200_ID_Info.json'
    gqa_id_info = json.load(open(gqa_200_ID_info))
    idx_to_predicate = gqa_id_info['rel_id_to_name']
    gqa_predicate_label = []
    for w in idx_to_predicate.values():
        if w != '__background__':
            gqa_predicate_label.append(w)
    
    # vg mapping
    vg_predicate_label = fine_predicates

    # vg-1800 mapping
    vg_1800_file = "/home/qifan/datasets/vg/1000/VG-dicts.json"
    vg_1800_info = json.load(open(vg_1800_file))
    vg_1800_predicate_label = []
    for label in vg_1800_info['predicate_to_idx']:
        vg_1800_predicate_label.append(label)

    prep_words = ['on','in','of','up','to','at']
    for raw_word in raw_predicates:
        mapping_dict[raw_word] = model_own.mapping_target(raw_word, raw_predicates, prep_words, device)

    json.dump(mapping_dict, open('utils_data/mapping/openworld_predicate_mapping_dict.json','w'))

