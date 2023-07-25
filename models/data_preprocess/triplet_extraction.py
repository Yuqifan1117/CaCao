from platform import node
from cv2 import split
import nltk, pandas as pd, numpy as np
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
import json

from tqdm import tqdm

dep_parser = CoreNLPDependencyParser(url='http://0.0.0.0:20001')
pos_tagger = CoreNLPParser(url='http://0.0.0.0:20001', tagtype='pos')

def convert_sentence (input_sent):
    # Parse sentence using Stanford CoreNLP Parser
    pos_type = pos_tagger.tag(input_sent.split())
    parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
    dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))
    return pos_type, parse_tree, dep_type

def multi_liaison (input_sent, output=['tagging','parse_tree','type_dep','spo','relation']):
    pos_type, parse_tree, dep_type = convert_sentence(input_sent)
    pos_sent = ' '.join([x[0]+'/'+x[1] for x in pos_type])
    # Extract subject, predicate and object
    subjects, adjective = get_subject(parse_tree)
    predicate = get_predicate(parse_tree)
    objects = get_object(parse_tree)
    # Generate the relations between subjects and objects 
    relation = get_relationship(parse_tree, subjects, objects)
    return relation

def get_subject (parse_tree):
    # Extract the nouns and adjectives from NP_subtree which is before the first / main VP_subtree
    subject, adjective = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        if s.label() == 'NP' and s.right_sibling() is not None and s.right_sibling().label() not in ['NP']:
            if s[-1].label() in ['NN','NNP','NNS','NNPS','PRP']:
                    for t in s.subtrees(lambda y: y.label() in ['NN','NNP','NNS','NNPS','PRP']):
                        output = t.pos()[0]
                        output_type = t.pos()[0][1]
                        if t.left_sibling() is not None and (t.left_sibling().label() in ['NN','NNP','NNS','NNPS','PRP'] or t.left_sibling().label().startswith('JJ')):
                            output = t.left_sibling().pos()[0] + output
                            output = ("-".join([output[0],output[2]]), output_type)
                        if t.right_sibling() is not None and t.right_sibling().label() in ['NN','NNP','NNS','NNPS','PRP$']:
                            output += t.right_sibling().pos()[0]
                            output = ("-".join([output[0],output[2]]), output_type)
                        # Avoid empty or repeated values
                        if output not in subject:
                            subject.append(output)
                    for t in s.subtrees(lambda y: y.label().startswith('JJ')):
                        if t.pos()[0] not in adjective:
                            adjective.append(t.pos()[0])
                    continue
            for x in s:
                if x in parse_tree.subtrees(lambda y: y.label() in ['NP']):
                    for t in x.subtrees(lambda y: y.label() in ['NN','NNP','NNS','NNPS','PRP']):
                        output = t.pos()[0]
                        output_type = t.pos()[0][1]
                        if t.left_sibling() is not None and (t.left_sibling().label() in ['NN','NNP','NNS','NNPS','PRP'] or t.left_sibling().label().startswith('JJ')):
                            output = t.left_sibling().pos()[0] + output
                            output = ("-".join([output[0],output[2]]), output_type)
                        if t.right_sibling() is not None and t.right_sibling().label() in ['NN','NNP','NNS','NNPS','PRP$']:
                            output += t.right_sibling().pos()[0]
                            output = ("-".join([output[0],output[2]]), output_type)
                        # Avoid empty or repeated values
                        if output not in subject:
                            subject.append(output)
                    for t in s.subtrees(lambda y: y.label().startswith('JJ')):
                        if t.pos()[0] not in adjective:
                            adjective.append(t.pos()[0])
    return subject, adjective

def get_predicate (parse_tree):
    # Extract the verbs from the VP_subtree
    predicate = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP' or x.label() == 'PP'):
        p_s_list = s.pos()
        object_list = list(s.subtrees(lambda y: y.label() == 'NP'))
        
        if len(object_list) == 0:
            continue
        
        for i in object_list[0].pos():
            p_s_list.remove(i)
        # print(p_s_list)
        predicate.append(p_s_list)
    return predicate

def get_object (parse_tree):
    # Extract the nouns from VP_NP_subtree
    objects, output = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP' or x.label() == 'PP'):
        for t in s.subtrees(lambda y: y.label() == 'NP'):
            for u in t.subtrees(lambda z: z.label() in ['NN','NNP','NNS','NNPS','PRP$']):
                output = u.pos()[0]
                output_type = u.pos()[0][1]
                if u.left_sibling() is not None and (u.left_sibling().label() in ['NN','NNP','NNS','NNPS','PRP'] or u.left_sibling().label().startswith('JJ')):
                    output = u.left_sibling().pos()[0] + output
                    output = ("-".join([output[0],output[2]]), output_type)
                elif u.right_sibling() is not None and u.right_sibling().label() in ['NN','NNP','NNS','NNPS','PRP$']:
                    output += u.right_sibling().pos()[0]
                    output = ("-".join([output[0],output[2]]), output_type)
                if output not in objects:
                    objects.append(output)
    return objects

def get_relationship(parse_tree, subjects, objects):
    relation = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP' or x.label() == 'PP'):
        p_s_list = s.pos()
        object_list = list(s.subtrees(lambda y: y.label() == 'NP'))
        
        if len(object_list) == 0:
            continue
        np_beginindex = p_s_list.index(object_list[0].pos()[0])
        # for i in object_list[0].pos():
        #     p_s_list.remove(i)
        p_s_list = p_s_list[:np_beginindex]
        str_p = []
        for p in p_s_list:
            str_p.append(p[0])

        subject = None
        predicate = None
        object = None
        for t in s.subtrees(lambda y: y.label() == 'NP'): 
            for u in t.subtrees(lambda z: z.label() in ['NN','NNP','NNS','NNPS','PRP$']):
                output = u.pos()[0]
                output_type = u.pos()[0][1]
                if u.left_sibling() is not None and (u.left_sibling().label() in ['NN','NNP','NNS','NNPS','PRP'] or u.left_sibling().label().startswith('JJ')):
                    output = u.left_sibling().pos()[0] + output
                    output = ("-".join([output[0],output[2]]), output_type)
                if u.right_sibling() is not None and u.right_sibling().label() in ['NN','NNP','NNS','NNPS','PRP$']:
                    output += u.right_sibling().pos()[0]
                    output = ("-".join([output[0],output[2]]), output_type)
                if output in objects and len(subjects) > 0:
                    # subject = subjects[subjects.index(output)-1]
                    n = objects.index(output)
                    while n > len(subjects) - 1:
                        n -= 1
                    subject = subjects[n]
                    predicate = ' '.join(str_p)
                    object = output
                    break
            break
        if subject != None and predicate != None and object != None:
            relation.append((subject[0],predicate,object[0]))
    return relation

if __name__ == '__main__':
    # instruction for start
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 20001 -timeout 30000


    # sentense1 = 'A clock that blends in with the wall hangs in a bathroom. '
    # sentense2 = 'A couple at the beach walking with their surf boards.'
    # sentense3 = "A yellow and black bird standing on and hanging with a bike rack."
    # relationships = multi_liaison(sentense3)
    regions = json.load(open("/home/qifan/datasets/vg/region_descriptions.json", 'r'))
    total_image_region_triplets = []
    for region in regions:
        image_region_triplets = dict()
        image_region_triplets['image_id'] = region['id']
        image_region_triplets['region_info'] = region['regions']
        for region_info in tqdm(image_region_triplets['region_info']):
            region_caption = region_info['phrase']
            region_triplets = []
            if region_caption is not '':
                relationships = multi_liaison(region_caption)
            else:
                relationships = []
            for r in relationships:
                region_triplets.append(r)
            region_info['region_triplets'] = region_triplets
        total_image_region_triplets.append(image_region_triplets) 
    json.dump(total_image_region_triplets, open('total_image_region_triplets.json','w'))  

    image_caption_dict = json.load(open("/home/qifan/datasets/coco/image_caption.json",'r'))
    coarse_relation = ['on', 'has', 'in', 'is', 'of', 'at', 'near', 'with', 'above', 'holding', 'behind', 'under', 'and', 'over', 'to', 'along', 'at', 'from', 'over', 'for', 'by', 'are', 'as', 'while'] 
    image_caption_triplet_dict = dict()
    n = 0
    for key in image_caption_dict.keys():
        image_info_novel = dict()
        image_info = image_caption_dict[key]
        enhanced_triplets = []
        for caption in tqdm(image_info['captions']):
            relationships = multi_liaison(caption)
            for relationship in relationships:
                if relationship[1] not in coarse_relation:
                    enhanced_triplets.append(relationship)
        image_info_novel['captions'] = image_info['captions']
        image_info_novel['image_file'] = image_info['image_file']
        image_info_novel['triplets'] = enhanced_triplets
        image_caption_triplet_dict[key] = image_info_novel
        print('contains relationships: ', len(enhanced_triplets))
        n += 1
        if n == 10000:
            break
    json.dump(image_caption_triplet_dict, open('image_caption_triplet.json','w'))