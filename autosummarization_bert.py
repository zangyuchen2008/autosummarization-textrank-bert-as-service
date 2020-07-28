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

# get doc and sens topic similarity
def get_topcs(doc,dictionary,lda_model,stop_words):
    tokens= list(filter(lambda x: (x not in stop_words) and (bool(re.findall('\w+', x))),jieba.cut(doc)))
    # return sorted(lda_model[dictionary.doc2bow(tokens)][0],key=lambda x : x[1],reverse=True)
    total_topcs = dict([(key, 0) for key in range(20) ])
    topcs =  lda_model[dictionary.doc2bow(tokens)][0]
    for a,b in topcs:
        total_topcs[a] = b 
    return list(total_topcs.values())
#  cross entropy
def cross_entropy(a,b):
    return sum([ -x * math.log2(y+0.000000000001) for x ,y in zip(a,b)])
# compare doc and sens topic similarity
def gettopic_similarity(doc,sens,dictionary,lda_model,stop_words):
    return cross_entropy(get_topcs(doc,dictionary,lda_model,stop_words),get_topcs(sens,dictionary,lda_model,stop_words))


# get extracted sentence, 
def get_extractin(doc,bc, stop_words,word_idf,threshhold,\
    key_para,pos_para,title='',trigger=True):
    spl_sens = split_sentence(doc)
    # sens and doc embedding
    sens_embedding = bc.encode(spl_sens)
    doc_embedding  = bc.encode([doc])

    # find key word in doc
    tfidfs = get_tfidfs(doc,word_idf,stop_words,5)
    sens_keyword_num = []
    for sen in spl_sens:
        words = list(jieba.cut(sen))
        keyword_num = sum([ 1 for w in words if w in dict(tfidfs)])
        sens_keyword_num.append(keyword_num)
    sens_embedding_ziped = list(zip(spl_sens,sens_embedding,sens_keyword_num,range(len(spl_sens))))
    
    sorted_sens=[]
    for sen, vec, keyword_num ,original_pos in sens_embedding_ziped:
        if title:
            # calculate title vec
            title_embedding  = bc.encode([title])
            # sort by cosine distance of doc, sentence , title, and multiply a keyword decrease based on keyword num, 
            score = (cosine(vec,doc_embedding[0]) * 0.5 + cosine(vec,title_embedding[0]) *0.5) * (1 - key_para * keyword_num)
            # add a sen postion decrease in doc(if sen is at begining or end, then give extra decrease)
            if (original_pos == 0) or (original_pos == len(sens_embedding_ziped)-1): score = score * (1 - pos_para)
        else:
            score = cosine(vec,doc_embedding[0]) * (1 - key_para * keyword_num)
            # add a sen postion decrease in doc(if sen is at begining or end, then give extra decrease)
            if (original_pos == 0) or (original_pos == len(sens_embedding_ziped)-1): score = score * (1 - pos_para)
        sorted_sens.append((sen, vec, keyword_num ,original_pos,score))

    # sort sens by weighted sens embedding vec score   
    sorted_sens = sorted(sorted_sens, key= lambda x: x[4])
    # select most relevent sens
    pops = sorted_sens[:threshhold]

    # sort by lda model(most relevent topics)
    # pops = sorted(pops,key=lambda x: x[5]) 
    
    # if start with special context,  move to top
    for index, p in enumerate(pops):
        if bool(re.search('^(\w+月\w+日\w*)|^(\w+日\w*)|^(\w+月\w*)|^(日前)|^([0-9]+年\w*)|^(.*报道\w*)',str.strip(p[0]))):
            pops.insert(0,pops.pop(index))
            break
    
    # if first sentece exists, this has higher priority, thus would overwrite the previouse one 
    for index, p in enumerate(pops):
        if p[3]==0 or bool(re.search('^(【.*((报道)|(记者)).*】)|^(\w*电\w*)',str.strip(p[0]))) :
            pops.insert(0,pops.pop(index))
            break

    # find most similar sentence next to each other(make sentence smooth)
    knn = []
    while pops:
        first = pops.pop(0)
        knn.append(first)
        pops = sorted(pops,key= lambda x : cosine(first[1],x[1]))
    print(np.array(knn).shape)
    return  knn
    
if __name__ == '__main__':
    # # load model
    # epoch_logger = EpochLogger()
    # model_path = os.path.join(os.path.abspath('./'),'word2vector_Model','word2vec.kv')
    # model = KeyedVectors.load(model_path,mmap='r')
    # # load sif
    # word_sifs =pickle.load(open('data\word_sifs.plk','rb'))
    # load stopwords
    stop_words = pickle.load(open('data\stop_words.plk','rb'))
    # load idf
    word_idf = pickle.load(open('data\word_idf.plk','rb'))

    # load lda model, and related dictionary
    # lda_model= LdaModel.load('lda_Model\lda_model')
    # dictionary=pickle.load(open('lda_Model\dictionary.plk','rb'))
    doc ='''这位曾在中国学医的阿富汗青年，本来梦想着治病救人，但最终，却在绝望中死去。

据美国有线电视新闻网（CNN）当地时间12月24日报道，赛义德·米尔瓦伊斯·鲁哈尼（Sayed Mirwais Rohani）是一位年轻的阿富汗医生，曾在中国求学。他本想逃离塔利班，却不料被困在澳大利亚的离岸移民拘留机构中。他在精神健康出现严重问题后，没有得到很好的治疗。

最终，10月15日，他32岁的生命在布里斯班城市酒店戛然而止。

他的律师乔治·纽豪斯（George Newhouse）表示，应该对此进行全面的调查，“他的家人想要弄清他们的儿子到底发生了什么事”。

“但更重要的是，要检查医疗保健系统的失败，这让一个身体健全的医生变得如此虚弱，看上去，他是自杀的。”

通过监控可以发现，鲁哈尼在去世的那一天，挎着一个黑包走进布里斯班市中心的一家酒店。他给自己的母亲打了最后一个电话，当时，他母亲正在澳大利亚陪他。第二天，警察就确认了他的死亡。

曾在中国学医的阿富汗青年 因受虐死于澳洲难民营

监控画面  本文图片均来自CNN

他的父亲表示，澳大利亚6年的移民拘留剥夺了自己儿子的所有希望，“当你把一只猫锁在房间里，然后关上门。猫只能这里那里到处跑，没有任何意义……移民局阻止他旅行、工作，也不允许他和家人团聚。”

11月1日，鲁哈尼在阿富汗喀布尔安葬，他在家人的陪伴下，又回到了这片出生的土地，可惜是以悲剧的方式。

曾在中国学医的阿富汗青年 因受虐死于澳洲难民营

11月1日下葬

留学中国，逃离阿富汗

赛义德·米尔瓦伊斯·鲁哈尼1987年出生在喀布尔，是一家6个孩子中的老二。据他父亲对CNN的描述，他“非常活跃、健康、喜欢学习。”

2001年，塔利班武装分子搜查了他们家，他的父亲塔桑瓦尔（Ahmad Tassangwal）逃离阿富汗，到了英国。当时，鲁哈尼14岁，但当父亲获准留在英国时，鲁哈尼已经成年了，他不能像母亲以及弟弟妹妹们一样，跟着父亲移民英国。

后来，鲁哈尼发现了中国泰山医学院一个他可以负担得起的留学项目，就来了中国学医，并于2012年毕业。

曾在中国学医的阿富汗青年 因受虐死于澳洲难民营

鲁哈尼毕业后，阿富汗的战争已经持续了十多年，他想去一个安全的国家。

他告诉在英国的父亲，“我在这个国家不安全，我要离开，去一个安全的国家。”

父亲告诉鲁哈尼，不要像自己当年一样想要尝试进入英国，因为他为了拿到签证浪费了很多年。父亲说，“我告诉他，如果你去澳大利亚，那是一个英语国家，是一个没什么经济问题的大国……你已经是个医生了，也许他们需要你。”

但没想到的是，当时，澳大利亚已经收紧了移民政策。2012年，为了应对涌入澳大利亚的移民船只，澳政府重新实施了一项离岸移民拘留政策，非法移民不能马上进入澳大利亚，只能先待在巴布亚新几内亚和瑙鲁的难民收容所，再慢慢予以“审核”。但澳大利亚法律并没有规定“审核”的明确期限。

精神状况恶化

2013年9月，鲁哈尼的船被澳大利亚边境部队拦截，作为一名“非法移民”，他被拘留，并被送往巴布亚新几内亚的马努斯岛，在那里，他待了4年。

鲁哈尼在收容所的室友称，鲁哈尼大部分时间都在看医学课本，每天读书十几个小时，说要提升自己，为了将来提升自己的职业水平。

他室友说，鲁哈尼会说6种语言，“他总是说他想开始自己的新生活，为穷人工作，帮助人们。”

可是，一心想帮助别人的他，自己却深陷绝望的境地。

在移民收容所里，鲁哈尼的精神状况恶化，精神失常却得不到很好的治疗。而他精神狂躁的行为，也让他沦为挨打的对象。

曾在中国学医的阿富汗青年 因受虐死于澳洲难民营

鲁哈尼在收容所被粗暴对待

2017年5月，塔桑瓦尔非常担心儿子的心理健康，他从英国飞到马努斯岛，试图把他带回家。但鲁哈尼没有旅行证件，没有澳大利亚和巴布亚新几内亚当局的许可，不能离开。

但在父亲离开后不久，鲁哈尼服用了过量的药物，被转移到了澳大利亚。医生诊断他患有双相情感障碍（bipolar affective disorder），这种情况会导致极端的情绪波动，在不治疗的情况下会恶化。

鲁哈尼和其他难民一起住在布里斯班的一所房子里，行动受到宵禁的限制，也没有得到有效的治疗。

鲁哈尼的父亲一直在尝试把儿子接走，2018年9月还飞往澳大利亚处理此事。但很可惜，直到儿子的生命结束，他都没有成功接走他。

澳大利亚处置难民的这种方式一直备受争议，这是因为被强制隔离的难民生活状况堪忧。2016年，联合国难民署的医学专家发现，被强制转移到巴布亚新几内亚和瑙鲁的难民中产生抑郁、焦虑和创伤后应激障碍的比例超过80%。根据非官方的澳大利亚边境死亡数据，到目前为止，已有13名难民死亡。

但是，澳大利亚政府多次为其离岸移民拘留政策辩护，称这是事关国家安全的问题。澳政府表示，拒绝难民在澳大利亚定居的机会降低了人们前往澳大利亚的动机，防止了海上死亡，并使澳大利亚的边境更加安全。'''

    # def test (doc ,title,threshhold):
    # print('get_extractin1:',''.join([a[0] for a in get_extractin1(doc,100,model,\
    #     word_sifs,stop_words,word_idf,3,0.15,0.2,dictionary,lda_model,'曾在中国学医的阿富汗青年 因受虐死于澳洲难民营',True)]))
    bc = BertClient()
    print('get_extractin1:',''.join([a[0] for a in get_extractin(doc,bc,\
        stop_words,word_idf,4,0.1,0.4,'曾在中国学医的阿富汗青年 因受虐死于澳洲难民营',True)]))