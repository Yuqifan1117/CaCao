import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor
from models.MLM.models import BertForPromptFinetuning
from models.MLM.MultiHeadAttention import TransformerLayer
from models.MLM.modeling_vit import ViTModel

from models.MLM.modeling_bert import BertForMaskedLM
from models.MLM.tokenization_bert_fast import BertTokenizerFast
from PIL import Image


class VisualBertPromptModel(nn.Module):
    def __init__(self, prefix_prompt_num, prompt_candidates, predicates_words, hidden_size=768, relation_type_count=31):
        super().__init__()
        self.prompt_num = prefix_prompt_num
        self.prefix_prompt = prompt_candidates[:prefix_prompt_num]
        self.word_table = predicates_words
        # prompt_ids means only tuning the position of prompt tokens
        # in visual-text prompt, text should pay attention in visual, thus need to train all of embedding layer
        prompt_ids = []
        for i in range(prefix_prompt_num):
            prompt_ids.append(i+1)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('/home/qifan/FG-SGG_from_LM/vit-base-patch32-224-in21k')
        self.visual_encoder = ViTModel.from_pretrained('/home/qifan/FG-SGG_from_LM/vit-base-patch32-224-in21k')
        self.transformerlayer = TransformerLayer(hidden_size=hidden_size, num_attention_heads=12, attention_probs_dropout_prob=0.1, intermediate_size=3072, hidden_dropout_prob=0.1, layer_norm_eps=1e-8)
        self.model = BertForMaskedLM.from_pretrained('/home/qifan/FG-SGG_from_LM/bert-base-uncased', prompt_ids)
        self.tokenizer = BertTokenizerFast.from_pretrained('/home/qifan/FG-SGG_from_LM/bert-base-uncased')
        self.prompt4re = BertForPromptFinetuning(self.model.config, relation_type_count, self.model.cls, self.word_table)
    def forward(self, batch_text, batch_img, weight=None, device='cuda:1', is_label=True):
        text_input = torch.Tensor([]).to(device)
        label = []
        attention_mask = torch.Tensor([])
        visual_prompts = torch.Tensor([]).to(device)
        mask_pos = []
        token_length = 30
        self.transformerlayer = self.transformerlayer.to(device)
        self.visual_encoder = self.visual_encoder.to(device)
        contexts = []
        for i in range(len(batch_text)):
            image = Image.open(batch_img[i]).convert("RGB")
            image_input = self.feature_extractor(images=image, return_tensors="pt").to(device)
            image_input['output_attentions'] = True
            outputs = self.visual_encoder(**image_input)
            cls_feature = outputs.pooler_output
            prompt4visual = torch.cat((torch.unsqueeze(cls_feature, dim=1), outputs.last_hidden_state[:,1:,:]), dim=1).to(device)
            visual_prompt = self.transformerlayer(prompt4visual)
            visual_prompts = torch.cat((visual_prompts, visual_prompt)) 
            visual_len = visual_prompt.shape[1]
            hard_prompt = 'The one relationship contain in the photo is '
            # text_prompt = hard_prompt + batch_text[i][0] + '[MASK]' + batch_text[i][2]
            text_prompt = batch_text[i][0] + ' [MASK] ' + batch_text[i][2]
            contexts.append(' '.join([batch_text[i][0], batch_text[i][1], batch_text[i][2]]))
            label_id = [self.word_table.index(batch_text[i][1])] if is_label else [-1]
            # text_prompt = 'The relationship is '+text_prompt
            input = self.tokenizer.encode(text_prompt, return_tensors="pt", add_special_tokens = False)
            # ['CLS']
            token = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map['cls_token'])]
            # prompt ids for soft_prompt
            prompt_str = []
            for prompt in self.prefix_prompt:
                prompt_str.append(prompt)
            for prompt in prompt_str:
                token_id = self.tokenizer.convert_tokens_to_ids(prompt)
                token.append(token_id)
            token = torch.cat((torch.Tensor(token), input[0]), dim=0)
            # ['SEP']
            token = torch.cat((token, torch.Tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map['sep_token'])]))).long()
            token = torch.unsqueeze(token, dim=0)
            # The image is [CLS][Visual-prompt vectors] with [SUB] and [OBJ]. The relationship is [SUB][MASK][OBJ]
            # [CLS][Visual-prompt vectors] [CLS] The relationship is [SUB][MASK][OBJ] [SEP]
            
            # attention mask
            token = add_zero(token, token_length)
            mask = torch.ones_like(token) - make_mask(token)
            visual_mask = torch.ones(1,visual_len)
            mask = torch.cat((visual_mask, mask), dim=1)
            
            attention_mask = torch.cat((attention_mask, mask)).long()
            # labels
            mask_idx = []
            for m in range(len(token[0])):
                if token[0][m] == 103:
                    mask_idx.append(m + visual_len)
            mask_pos.append(mask_idx)
            token = token.to(device)
            text_input = torch.cat((text_input, token)).long().to(device)
            label.append(label_id)
        label = torch.Tensor(label).long().to(device)
        attention_mask = attention_mask.to(device)
        self.model = self.model.to(device)
        # mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
        # position_ids = torch.arange(attention_mask.shape[1]).expand((1, -1)).to(device)
        output = self.model(input_ids=text_input, labels=label, visual_prompts=visual_prompts, attention_mask=attention_mask)
        mask_pos = torch.Tensor(mask_pos).long()
        self.prompt4re = self.prompt4re.to(device)
        weight = torch.Tensor(weight).to(device) if weight is not None else None
        if not is_label:
            label = None
        final = self.prompt4re(output[1], mask_pos=mask_pos, labels=label, contexts=contexts, weight=weight, device=device)
        # logits = output.logits
        return final, label if label is not None else final
    def mapping_target(self, predicted_rel, target_words, prep_words, device='cuda:0'):
        embedding_model = self.model.bert.get_input_embeddings()
        if predicted_rel not in prep_words:
            predicted_keyword_split = predicted_rel.split(' ')
            for prep in prep_words:
                if prep in predicted_keyword_split:
                    predicted_keyword_split.remove(prep)
            predicted_rel = ' '.join(predicted_keyword_split)
        text_predict_prompt = self.tokenizer.encode(predicted_rel, return_tensors="pt", add_special_tokens = False).to(device)
        predicted_embedding_token = embedding_model(text_predict_prompt)
        predicted_embedding = torch.sum(predicted_embedding_token, dim=1)
        predicted_embedding /= predicted_embedding_token.shape[1]
        all_keywords_w2v_list = []
        for target_rel in target_words:
            with torch.no_grad():
                if target_rel not in prep_words:
                    original_target_rel = target_rel
                    keyword_split = target_rel.split(' ')
                    for prep in prep_words:
                        if prep in keyword_split:
                            keyword_split.remove(prep)
                    target_rel = ' '.join(keyword_split)
                # additional hierarchy structure for mapping
                text_target_prompt = self.tokenizer.encode(target_rel, return_tensors="pt", add_special_tokens = False).to(device)
                embedding_token = embedding_model(text_target_prompt)
                total_embedding = torch.sum(embedding_token, dim=1)
                total_embedding /= embedding_token.shape[1]
                all_keywords_w2v_list.append((original_target_rel, total_embedding))
        target_worddic = dict(all_keywords_w2v_list)

        similarity = [torch.cosine_similarity(predicted_embedding, target_worddic[w[0]], dim=1) for w in all_keywords_w2v_list]
        similarity = torch.cat(similarity, dim=-1)
        top_score = torch.topk(similarity, k=3)
        top_score_index = top_score[1]
        # max_score = max(similarity)
        # max_score_index = similarity.index(max_score)
        result_words = []
        for i in top_score_index:
            result_words.append(all_keywords_w2v_list[i][0])
        return result_words
        # return all_keywords_w2v_list[max_score_index][0]
            

def construct_input(subject, predicate, object, tokenizer, prefix_prompt):
    text_prompt = subject + predicate + object
    input = tokenizer.encode(text_prompt, return_tensors="pt", add_special_tokens = False)
    # ['CLS']
    token = [tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['cls_token'])]
    # prompt ids
    prompt_str = []
    for prompt in prefix_prompt:
        prompt_str.append(prompt)
    for prompt in prompt_str:
        token_id = tokenizer.convert_tokens_to_ids(prompt)
        token.append(token_id)
    token = torch.cat((torch.Tensor(token), input[0]), dim=0)
    # ['SEP']
    token = torch.cat((token, torch.Tensor([tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['sep_token'])]))).long()
    token = torch.unsqueeze(token, dim=0)     
    return token   
def add_zero(token, token_length):
    zero = torch.Tensor([[0] * (token_length-token.shape[1])])
    return torch.cat((token, zero), dim=1)
def make_mask(feature):
    return (feature[0] == 0).long().unsqueeze(0)

    
