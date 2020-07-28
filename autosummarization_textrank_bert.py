# coding=utf8
import gensim
import pickle
from gensim.models import Word2Vec,LdaModel,KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import jieba
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math
# from sen2vec import sentence_to_vec
import os
import warnings
import networkx as nx 
from bert_serving.client import BertClient
warnings.filterwarnings("ignore")

# split sentence
def split_sentence(sentence):
    result= re.split('[。?？!！\r\n]',sentence)
    senswith_punct=[]
    # recover punctuation
    for index , s in enumerate(result):
        backward_length = list(map(len,result[:index+1]))
        total_length = sum(backward_length) + index
        if total_length< len(sentence):
            notation = sentence[total_length]
            senswith_punct.append(s + notation)
    result = list(filter(lambda x: len(x)>3,senswith_punct))
    result_cleaned = []
    while result:
        sen = result.pop()
        # remove duplicated sentence, remove subtitle in the document (which has no puctuation)
        if np.sum([str.strip(sen) == str.strip(s) for s in result])==0 and \
            bool(re.search('[。?？!！]',sen)): result_cleaned.insert(0,sen) 
        else: continue
    return result_cleaned

# get sentence tfidf
def get_tfidfs(sentece,word_idf,stop_words,threshhold):
    words_filtered = list(filter(lambda x :bool(re.findall('\w+',x)) and (x not in stop_words) , jieba.cut(sentece)) )
    sen_word_count = Counter(words_filtered)
    tfidfs = []
    for a, b in sen_word_count.most_common():
        if a in word_idf:
            tfidfs.append((a,  b/len(words_filtered) * word_idf[a]))
        else :
            tfidfs.append((a,  b/len(words_filtered)))
    return sorted(tfidfs,key=lambda x: x[1],reverse=True)[:threshhold]

# get textrank result
def get_textrank(sens_embedding, sens,tfidfs,para_title, para_keyword,para_fisrtsen):
    # keyword overlap
    sens_keywords = {}
    key_words = [a for a,b in tfidfs]
    for index, sen in enumerate(sens):
        words = list(jieba.cut(sen))
        sens_keyword = [w for w in words if w in key_words]
        sens_keywords[index] = sens_keyword
    # create graph
    G= nx.Graph()
    edges = []
    for i , v1 in enumerate(sens_embedding):
        for j, v2 in enumerate(sens_embedding[i+1 :]):
            com_keyword_num = len(set(sens_keywords[i]) &  set(sens_keywords[i+j+1]))
            # we decrease cosin distance between sens based on commmon key word number
            score = cosine(v1,v2)*(1- com_keyword_num*para_keyword) 
            if i ==0:
                score = score * para_title # weight for relation with title 
                edges.append((i,j+i+1,score))
            else:
                edges.append((i,j+i+1,score))
    G.add_weighted_edges_from(edges)
    # pagerank
    page_rank = nx.pagerank(G,weight='weight')
    # weight first sentense
    page_rank[1] = page_rank[1] * para_fisrtsen
    # sorted based on ranking values
    result = sorted(zip(page_rank.items(),sens,sens_embedding),key=lambda x: x[0][1])
    return result, G

def autosummation(title,doc,bc,word_idf,stop_words):
    # remove period, question mark etc. punctuation in case affect sentence splitting(title splitted into several parts)
    title = re.sub('[。?？!！\r\n]','**',title)
    # 分句
    spl_sens = split_sentence(str.strip(title)+ '。'+str.strip(doc))
    # sens_embedding1 , spl_sens_cleared = sentence_to_vec(spl_sens,100,model,word_sifs,stop_words,)
    sens_embedding = bc.encode(spl_sens)
    # get document keywords via tfidf
    tfidfs = get_tfidfs(doc,word_idf,stop_words,10)
    # get textrank
    result ,G = get_textrank(sens_embedding,spl_sens,tfidfs,0.5,0.05,0.8)
    # sort based on original sequence in document
    key_sentences = sorted(result[:4],key= lambda x: x[0][0])
    return ''.join([b for a,b,c in key_sentences])



if __name__ == '__main__':
    bc = BertClient()
    # load stopwords
    stop_words = pickle.load(open('data\stop_words.plk','rb'))
    # load idf
    word_idf = pickle.load(open('data\word_idf.plk','rb'))
  
    title= '''近六分之一国土雨量超200毫米！这么多的雨都是哪来的？'''
    doc = '''
    湖北鄂州机场作为湖北国际物流核心枢纽项目，其文物保护项目已竣工通过验收。
    作为继三峡、南水北调工程建设后湖北省最大的文物保护项目，该项目获得丰硕成果。
    在各方的配合支持下，湖北省文化和旅游厅组织省文物考古研究所、
    鄂州市博物馆等多支考古队伍，对机场建设核心区域内27处文物点，
    实施了近三个月的勘探发掘，共勘探面积近90万平方米，发掘面积1.26万平方米。
    据不完全统计，共发掘墓葬183座、窑址20座、灰坑10处，出土青瓷、陶、铜、铁、
    滑石器等各类重要文物800余件，标本近万件。
    '''
   
    # summation output
    result = autosummation(title,doc,bc,stop_words,word_idf)
    print(result)
