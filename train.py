# -*-coding:utf-8 -*-

import pymysql
from scrapy.selector import Selector
from scrapy.selector import HtmlXPathSelector
import re
import codecs
import jieba
import numpy as np
import gensim
from gensim.models import word2vec
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model, load_model
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Lambda
from keras.models import Sequential
import random
from random import shuffle
from keras.callbacks import ModelCheckpoint
from MANM import create_model

import keras

conn = pymysql.connect(host='localhost',
                       port=3306,
                       user='root',
                       passwd='XXXXXXX',
                       db='maimang_wenben',
                       charset='utf8')

from textrank4zh import TextRank4Keyword, TextRank4Sentence, Segmentation


def segment(text, lower=True):
    """对一段文本进行分词，返回list类型的分词结果

    Keyword arguments:
    lower                  -- 是否将单词小写（针对英文）
    use_stop_words         -- 若为True，则利用停止词集合来过滤（去掉停止词）
    use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。    
    """
    jieba_result = jieba.cut(text)

    word_list = [word for word in jieba_result if len(word) > 0]

    if lower:
        word_list = [word.lower() for word in word_list]

    return word_list


# 分句
def get_sentences(doc, comma_cut=False):
    line_break = re.compile('[\r\n]')
    if comma_cut == False:
        delimiter = re.compile(u'(。？！；)')  # 保留分隔符
    else:
        delimiter = re.compile(u'[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences


def get_content_from_sql(conn):
    # try:
    text_sentence_list_all = []
    doc_list_all = []
    cursor = conn.cursor()
    effect_row = cursor.execute("select distinct id,name,digest,content from seo_news_bot")
    data_row = cursor.fetchall()
    card_dic = {}

    for item in data_row:
        id = item[0]
        sentence_cut_list = []
        sel = Selector(text=item[3])
        rawCont = sel.xpath("//p")
        print("####################################################################################################")
        text = ""
        # print(rawCont)
        for cont in rawCont:
            # print("ddddd")
            sentence_list = []
            for nn in cont.xpath('text()').extract():

                text += nn.strip()
                text += '\n'
                cut_sentence_list = get_sentences(nn)

                for s in cut_sentence_list:
                    cut = segment(text=s)

                    sentence_list.append(cut)

        text_sentence_list_all.append(sentence_list)
        d_cut = segment(text=text)
        doc_list_all.append(d_cut)

    cursor.close()
    # conn.close()
    return doc_list_all, text_sentence_list_all
    # except:
    # print ("error in func <get_card_from_sql>")
    # return [],[]


doc_list_all, text_sentence_list_all = get_content_from_sql(conn)

"""
# 词向量
cop = doc_list_all
word2vec_model = word2vec.Word2Vec(cop, size=64 ,min_count = 0)
word2vec_model.wv.save('word2vec')
"""
word2vec_model = gensim.models.KeyedVectors.load('../corpus_process/word2vec')


def get_positive_input_and_output(text_sentence_list_all):
    max_num = 0
    inputs_and_targets = []
    for text_sentence in text_sentence_list_all:
        sentence_num = len(text_sentence)
        for i in range(sentence_num - 1):
            if min(len(text_sentence[i]),len(text_sentence[i + 1]))>0 and max(len(text_sentence[i]),len(text_sentence[i + 1])) < 200:
                max_num = max(max_num, len(text_sentence[i]), len(text_sentence[i + 1]))
                inputs_and_targets.append(((text_sentence[i], text_sentence[i + 1]), 1))

    print("pos_len:%d" % (max_num))
    return inputs_and_targets


positive_inputs_and_targets = get_positive_input_and_output(text_sentence_list_all)


def get_negative_input_and_output(text_sentence_list_all, positive_inputs_and_targets, multiple=1):
    max_num = 0
    inputs_and_targets = []
    # 所有句子展开
    sentence_list_all = [y for x in text_sentence_list_all for y in x]
    positive_cups = [cup for cup, _ in positive_inputs_and_targets]
    sentence_all_nums = len(sentence_list_all)
    negative_nums = len(positive_inputs_and_targets) * multiple

    for i in range(negative_nums):
        s_idx = random.randint(0, sentence_all_nums - 1)
        d_idx = random.randint(0, sentence_all_nums - 1)
        if (sentence_list_all[s_idx], sentence_list_all[d_idx]) not in positive_cups and max(len(sentence_list_all[s_idx]),len(sentence_list_all[d_idx])) < 200:
            max_num = max(max_num, len(sentence_list_all[s_idx]), len(sentence_list_all[d_idx]))
            inputs_and_targets.append(((sentence_list_all[s_idx], sentence_list_all[d_idx]), 0))

    print("neg_len:%d" % (max_num))
    return inputs_and_targets


negative_input_and_output = get_negative_input_and_output(text_sentence_list_all=text_sentence_list_all,
                                                          positive_inputs_and_targets=positive_inputs_and_targets)

input_and_output = positive_inputs_and_targets + negative_input_and_output

print(len(input_and_output))
"""
N_UNITS = 256
BATCH_SIZE = 60
EPOCH = 200
EMBEDDING_SIZE = 64
MAX_STEP = 435
"""

N_UNITS = 100
BATCH_SIZE = 60
EPOCH = 5
EMBEDDING_SIZE = 64
MAX_STEP = 200

q_embedding_input = np.zeros((len(input_and_output), MAX_STEP, EMBEDDING_SIZE))
p_embedding_input = np.zeros((len(input_and_output), MAX_STEP, EMBEDDING_SIZE))

label = np.zeros(len(input_and_output))

for idx, ((first_words_list, last_words_list), mask) in enumerate(input_and_output):
    for i in range(len(first_words_list)):
        q_embedding_input[idx][i] = word2vec_model[first_words_list[i]]

    for j in range(len(last_words_list)):
        p_embedding_input[idx][j] = word2vec_model[last_words_list[j]]

    label[idx] = mask


def acc_pred(y_true, y_pred):
    y_true_bool = y_true == 1
    y_pred_bool = y_pred >= 0.5
    t = y_true_bool == y_pred_bool
    return K.mean(K.cast(y_pred_bool, dtype='float64'))


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)




model = create_model(EMBEDDING_SIZE,N_UNITS,MAX_STEP,BATCH_SIZE)

#查看模型结构
plot_model(to_file='model.png',model=model,show_shapes=True)


model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])


# checkpoint
filepath = "model/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# model.fit(embedding_input, label, batch_size=BATCH_SIZE, epochs=EPOCH,validation_split=0.2,shuffle=True)
model.fit([q_embedding_input,p_embedding_input], label, batch_size=BATCH_SIZE, epochs=EPOCH,validation_split=0.2,shuffle=True)
model.save("model.test")


