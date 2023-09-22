import json
from cv2 import split
from sklearn import metrics
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
import scipy
import torch.nn.functional as F
from models.MLM.utils import fineTuningDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import pandas as pd 

def SSE_clu(v1,v2):
    return sum(np.power(v1 - v2,2)) 

class WordsCluster(object):
    def __init__(self, embedding_file, sim_threshold=0.55, del_sim_threshold=0.2, ignore_keywords=[], prep_words=[]):
        self.sim_threshold = sim_threshold
        self.del_sim_threshold = del_sim_threshold
        self.model = BertModel.from_pretrained(embedding_file)
        self.tokenizer = BertTokenizerFast.from_pretrained(embedding_file)
        self.embedding_model = self.model.get_input_embeddings()
        self.ignore_keywords = ignore_keywords
        self.prep_words = prep_words
        self.unexist = set()
    # 初始化embedding信息
    def initial_embedding_info(self, keywords):
        all_keywords_w2v_list = []
        all_keywords_embeddings = torch.tensor([])
        # get embedding of each predicate by the mean of GLOVE vectors of all triplets 
        for keyword in keywords:
            with torch.no_grad():
                predicate_embedding = []
                for triplet in keywords[keyword]:
                    total_embedding = []
                    for w in triplet:
                        input = self.tokenizer.encode(w, return_tensors="pt", add_special_tokens = False)
                        embedding_token = self.embedding_model(input)
                        word_embedding = torch.mean(embedding_token, dim=1)
                        total_embedding.append(word_embedding)
                    total_embedding = torch.cat(total_embedding, dim=0)
                    total_embedding = torch.sum(total_embedding, dim=0)
                    total_embedding = total_embedding / len(triplet)
                    predicate_embedding.append(total_embedding)
                predicate_embedding = torch.stack(predicate_embedding, dim=0)
                predicate_embedding = torch.mean(predicate_embedding, dim=0).unsqueeze(0)
                all_keywords_w2v_list.append((keyword, predicate_embedding))
                all_keywords_embeddings = torch.cat((all_keywords_embeddings, predicate_embedding), dim=0)
        self.words_w2v_dic = dict(all_keywords_w2v_list)
        return all_keywords_embeddings
 
    # 获取类标
    def get_class_represent_word(self, collection):
        if len(collection) == 0:
            return None
        if len(collection) == 1:
            return collection[0]
        # 计算模平均，求模平均w2v相似度最高词语
        sim_list = [torch.cosine_similarity(key_w2v, torch.mean(torch.stack(collection['words_w2v']), dim=0), dim=1).item() for key_w2v in collection['words_w2v']]
        max_sim = max(sim_list)
        if len(sim_list)==1:
            max_sim = sim_list[0]
        max_sim_index = sim_list.index(max_sim)
        represent_word = collection["words"][max_sim_index]
        represent_word_sim = max_sim
 
        return represent_word, represent_word_sim
 
    # 过滤
    def filt_noise_words(self, collection):
        delete_noise_opinion_words_indexes = []
        scores = [np.mean(np.dot(collection['words_w2v'], x)) for x in collection['words_w2v']]
        # 根据阈值过滤类中相似度较小词
        for i in range(len(scores)):
            if scores[i] <= self.del_sim_threshold:
                delete_noise_opinion_words_indexes.append(i)
        collection['words_w2v'] = [x for (i, x) in enumerate(collection['words_w2v']) if
                                   i not in delete_noise_opinion_words_indexes]
        collection['words'] = [x for (i, x) in enumerate(collection['words']) if
                               i not in delete_noise_opinion_words_indexes]
    # t-sne 可视化
    def t_sne_kmeans(self, words_w2v_embeddings):
        vecArr = np.array(words_w2v_embeddings) 
        tsneData = TSNE().fit_transform(vecArr)

        #开始进行可视化
        f = plt.figure(figsize=(10,10))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(tsneData[:,0], tsneData[:,1])
        plt.xlim(-50,50)
        plt.ylim(-50,50)
        ax.axis('off')
        ax.axis('tight')
        plt.savefig('tsne.png')

    # 计算欧拉距离
    def calcDis(self, dataSet, centroids, k):
        clalist=[]
        for data in dataSet:
            diff = np.tile(data, (k, 1)) - centroids  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
            squaredDiff = diff ** 2     #平方
            squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
            distance = squaredDist ** 0.5  #开根号
            clalist.append(distance) 
        clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
        return clalist

    # 计算质心
    def classify(self, dataSet, centroids, k):
        # 计算样本到质心的距离
        clalist = self.calcDis(dataSet, centroids, k)
        # 分组并计算新的质心
        minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
        newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
        newCentroids = newCentroids.values
    
        # 计算变化量
        changed = newCentroids - centroids
    
        return changed, newCentroids


    def cluster(self, keywords, num_clusters, sim_threshold=None):
        '''
            keywords词语聚类
            :param keywords_dic:
            :return:
        '''
        result = []
        if sim_threshold is None:
            sim_threshold = self.sim_threshold
        # we collect all keywords embedding in triplet-level, and get the mean of all relevant triplets as the embedding
        all_keyfeatures = self.initial_embedding_info(keywords)
        st = StandardScaler()
        # all_keyfeatures = st.fit_transform(all_keyfeatures.numpy())
        sk_kmeans = KMeans(n_clusters=num_clusters)
        result_list = sk_kmeans.fit(all_keyfeatures)
        centroids = result_list.cluster_centers_
        closest_centroids_ids = result_list.labels_
        # centroids, closest_centroids_ids = self.train(all_keyfeatures.numpy(), num_clusters, max_iterations=10)
        # find the represented word
        cluster_dict = dict()
        for i, centroid in enumerate(centroids):
            similarity = -1
            for k in self.words_w2v_dic:
                cur_similarity = torch.cosine_similarity(torch.tensor(centroid), self.words_w2v_dic[k], dim=-1)
                if cur_similarity.item() > similarity:
                    similarity = cur_similarity
                    centroid_label = k
            cluster_dict[str(i)] = dict()
            cluster_dict[str(i)]["represent_word"] = centroid_label
            cluster_dict[str(i)]["words"] = []
            cluster_dict[str(i)]["represent_word_sim"] = similarity.item()
            for m, k in enumerate(self.words_w2v_dic):
                if closest_centroids_ids[m] == i:
                    cluster_dict[str(i)]["words"].append(k)
        return cluster_dict

 

    def upgrade_cluster(self, keywords, collection_words_list=[], sim_threshold=None):
        if not sim_threshold:
            sim_threshold = self.sim_threshold
        for i in range(7, min(1, int(sim_threshold * 10) - 3), -1):
            collections, un_seg_words = self.cluster(keywords, collection_words_list, i * 0.1)
            keywords = un_seg_words
            collection_words_list = [x['words'] for x in collections]
        return collections, un_seg_words
 
    def sim(self, w1, w2):
        if w1 in self.embedding_model.unexist or w2 in self.unexist:
            return -1
        w1_w2v = self.embedding_model.get_word_embedding(w1)
        w2_w2v = self.embedding_model.get_word_embedding(w2)
        return np.dot(w1_w2v, w2_w2v) 
    def get_embedding(self, triplet):
        total_embedding = []
        for w in triplet:
            input = self.tokenizer.encode(w, return_tensors="pt", add_special_tokens = False)
            embedding_token = self.embedding_model(input)
            word_embedding = torch.mean(embedding_token, dim=1)
            total_embedding.append(word_embedding)
        total_embedding = torch.cat(total_embedding, dim=0)
        total_embedding = torch.mean(total_embedding, dim=0)
        return total_embedding 
    def train(self, data, num_clusters, max_iterations):
        # data precess
        centroids=self.centroids_init(data, num_clusters)
        # print(data[0])
        #2.开始训练
        num_examples=data.shape[0]
        closest_centroids_ids=np.empty((num_examples,1))
        for i in range(max_iterations):
            # print('current iterations: ', i)
            # 得到当前每个样本到k个中心点的距离，找最近的
            closest_centroids_ids=self.centroids_find_closest(data,centroids)   
            #进行中心点位置更新
            centroids=self.centroids_compute(data,closest_centroids_ids,num_clusters)
        # print(centroids)
        # print(closest_centroids_ids)
        return centroids, closest_centroids_ids
    def centroids_init(self, data, num_clusters):
        num_examples=data.shape[0]
        random_ids=np.random.permutation(num_examples) # shuffle the id and select the random centroids
        centroids=data[random_ids[:num_clusters],:]
        return centroids
    def centroids_find_closest(self,data,centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples,1))
        for example_index in range(num_examples) :
            distance = np.zeros((num_centroids,1))
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index,:] - centroids[centroid_index,:]
                distance[centroid_index] = np.sum(distance_diff**2)
                closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids
    def centroids_compute(self, data, closest_centroids_ids, num_clusters):
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters,num_features))
        for centroid_id in range(num_clusters) :
            closest_ids = np.where(closest_centroids_ids == centroid_id)[0]
            centroids[centroid_id] = np.mean(data[closest_ids,:],axis=0)
        return centroids


    def evaluate_func(self, keywords):
        all_keyfeatures = self.initial_embedding_info(keywords)
        # ls_k = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
        ls_k = range(10,562,10)
        # ls_k = range(2,50)
        ls_sil = []
        ls_ch = []
        ls_elbows =[]
        ls_gs = []
        st = StandardScaler()
        all_keyfeatures = st.fit_transform(all_keyfeatures.numpy())
        for i in ls_k:
            ls_elbow = []
            
            # centroids, closest_centroids_ids = self.train(all_keyfeatures, i, max_iterations=30)
            sk_kmeans = KMeans(n_clusters=i)
            
            result_list = sk_kmeans.fit(all_keyfeatures)
            res2 = result_list.cluster_centers_
            res1 = result_list.labels_
            # res1 = closest_centroids_ids.astype(int).T.tolist()[0]###输出的是聚类类比 closest_centroids_ids
            # res2 = centroids##输出的是聚类中心 centroids
            # normalize all keyfeatures
            # ls_gs.append(self.gap(all_keyfeatures, i))
            for j in range(len(res1)):
                choose_label = res2[int(res1[j]), :]
                sse = SSE_clu(all_keyfeatures[j, :], choose_label)##肘方法
                ls_elbow.append(sse)
            # print(ls_elbow)
            ls_sil.append(metrics.silhouette_score(all_keyfeatures,res1))###轮廓系数
            ls_ch.append(metrics.calinski_harabasz_score(all_keyfeatures,res1))###CH值
            ls_elbows.append(sum(ls_elbow))
        return ls_elbows,ls_sil,ls_ch,ls_gs
    


    def sum_distance(self, data, k):
        model = KMeans(n_clusters=k)
        result_list = model.fit(data)
        res1 = result_list.labels_
        res2 = result_list.cluster_centers_
        
        disp = 0
        for m in range(data.shape[0]):
            disp += np.linalg.norm(data[m] - res2[res1[m]], axis=0)
        return disp

    def gap(self, data, k):
        shape = data.shape
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(np.diag(tops - bots))    
        rands = scipy.random.random_sample(size=(shape[0], shape[1]))
        rands = rands * dists + bots
        disp = self.sum_distance(data, k)
        refdisps = self.sum_distance(rands, k)
        gap = np.lib.scimath.log(np.mean(refdisps)) - np.lib.scimath.log(disp)
        return gap

    def monte_carlo(self, keywords, epochs=10):
        matx_elbows = np.mat(np.zeros((epochs, 56)))
        matx_sil = np.mat(np.zeros((epochs, 56)))
        matx_ch = np.mat(np.zeros((epochs, 56)))
        matx_gs = np.mat(np.zeros((epochs, 56)))
        for i in range(epochs):
            Repoch = self.evaluate_func(keywords)
            matx_elbows[i, :] = Repoch[0]
            matx_sil[i, :] = Repoch[1]
            matx_ch[i, :] = Repoch[2]
            # matx_gs[i, :] = Repoch[3]

        mean_elbows = matx_elbows.sum(axis=0) / epochs
        mean_sil = matx_sil.sum(axis=0) / epochs
        mean_ch = matx_ch.sum(axis=0) / epochs
        # matx_gs = matx_gs.sum(axis=0) / epochs
        st = StandardScaler()
        # mean_ch = st.fit_transform(mean_ch)
        # print(mean_ch)
        # mean_ch = mean_ch / max(mean_ch.tolist()[0])

        print('SSE',mean_elbows.tolist()[0])
        print('轮廓系数',mean_sil.tolist()[0])
        print('Norm CH值',mean_ch.tolist()[0])
        # print('Gap Statistic',matx_gs.tolist()[0])
        # plt.figure(figsize=(15,8))
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        # X = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
        X = range(10,562,10)
        ax1.plot(X, mean_elbows.tolist()[0], marker='o', label='Elbow')
        ax2.plot(X, mean_sil.tolist()[0], 'r', marker='*', label='Silhouette Coefficient')
        # ax2.plot(X, mean_ch.tolist()[0], 'g', marker='*', label='CH norm')
        # ax2.bar(X, matx_gs.tolist()[0], label='Gap Statistic')
        ax1.set_ylabel('SSE', fontsize=20)
        ax1.set_xlabel('K', fontsize=20)
        ax2.set_ylabel('Value', fontsize=20)
        ax1.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        ax1.legend(loc='lower left', fontsize=20)
        ax2.legend(loc='upper right',fontsize=20)
        # plt.show()
        plt.savefig('centroids_4.png')
  
if __name__ == '__main__':
    vg_dataset = fineTuningDataset('datasets/image_caption_triplet_all.json',"/home/qifan/datasets/coco/train2014/",'train')
    # train_dataset = fineTuningDataset('gqa_triplets.json',"/home/qifan/datasets/GQA/images/",'train')
    data_loader = DataLoader(vg_dataset, batch_size=8, shuffle=True)
    predicate_words = vg_dataset.predicates_words
    predicate_dict = dict()
    for p in predicate_words:
        predicate_dict[p] = [[p]]
    for triplet_info in vg_dataset.triplets:
        triplet = triplet_info['triplet']
        predicate_dict[triplet[1].lower()].append(triplet)
    prep_words = []
    ignore_words = []
    kmeans = WordsCluster('/home/qifan/FG-SGG_from_LM/bert-base-uncased', ignore_keywords=ignore_words, prep_words=prep_words)
    # kmeans.monte_carlo(predicate_dict)
    # using sim_threshold to initilize the number of clusters for total classes
    # cluster_dict = kmeans.cluster(predicate_dict, num_clusters=230, sim_threshold=0.7)
    # json.dump(cluster_dict, open('utils_data/cluster/CaCao_all_cluster_dict_07.json', 'w'))
    # using sim_threshold to initilize the number of clusters for target classes
    cluster_dict = kmeans.cluster(predicate_dict, num_clusters=39, sim_threshold=0.7)
    json.dump(cluster_dict, open('utils_data/cluster/CaCao_map50_dict_07.json', 'w'))

