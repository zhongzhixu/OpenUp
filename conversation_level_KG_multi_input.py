import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM, Bidirectional,Input,concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from gensim.models import word2vec
import gensim.models.keyedvectors as word2vec
from sklearn.metrics import classification_report
import opencc
import re
import emoji
from copy import deepcopy

# rewrite
KG_embedding = True
multi_input = True
load_weights = True
randstate = 24 #should be 24

filepath = './newdata_add_counsellor/'
#filename = 'training_cases_1130.csv'
filename = 'training_cases_10_10_V3.csv'
df = pd.read_csv(filepath+filename)


df.dropna(subset=["conversationId"],inplace=True)

cc = opencc.OpenCC('s2hk')
def to_hk(text):
    return cc.convert(text)
df['helpseeker_msg'] = df['helpseeker_msg'].str.replace(
    r'[\x00-\x7F]+', ' ').astype(str).apply(
    lambda x: ''.join(c for c in x if c not in emoji.UNICODE_EMOJI) ).str.strip()
df['helpseeker_msg'] = df['helpseeker_msg'].apply(to_hk)

#%%
if filename == 'merged_1206.csv':
    
    train = df[df['helpseeker_msg'].map(len)>5] #---
    train.riskLevel.value_counts()
    train_text = train['helpseeker_msg'].to_list()

if ( ('training_cases_10_10_V' in filename) or filename == 'training_cases_1130.csv'):
    # sample
    train = df[df['helpseeker_msg'].map(len)>4]
    if ('training_cases_10_10_V' in filename): 
        train = train.drop(columns='riskLevel').rename(columns={'rlfill':'riskLevel'})
    if filename == 'training_cases_1130.csv':
        train = train.rename(columns={'rlfill':'riskLevel'})
    train.riskLevel.value_counts()
    train_high = train[train['riskLevel'] > 1]
    train_high.riskLevel=train_high.riskLevel.replace(3,1)
    train_high.riskLevel=train_high.riskLevel.replace(2,1)
    train_high.riskLevel.value_counts()
    train_low = train[train['riskLevel'] <= 1]
    low = train_low[train_low['riskLevel'] == 0]
    medium = train_low[train_low['riskLevel'] == 1]
    medium['riskLevel'] = medium['riskLevel'].replace(1,0)
    
    ones, zeros_medium, zeros_low = 670, 800, 4560 #------should be 670, 800, 4560
    zeros = zeros_medium + zeros_low
    try: high_s = train_high.sample(ones,random_state = randstate)
    except: 
        ones = train_high.shape[0]
        high_s = train_high
    medium_s = medium.sample(zeros_medium,random_state = randstate)
    low_s = low.sample(zeros_low,random_state = randstate)
    
    data = pd.concat([high_s,medium_s,low_s])
    data = data.sort_values(by='riskLevel',ascending=False)
    data.index = range(data.shape[0])
    data['is_exit'] = data['exit_group'] == data ['anno_group']
    
    
    ###
    #data.loc[278,'counsellor_msg'] = 'Hello 你好. 我係當值輔導員呀. 請問點稱呼你?你好啊,我叫kam今日你上嚟呢度有D咩想同我傾啊?'
    #data.loc[278,'helpseeker_msg'] = '你好,我係Yannis,心情唔好好攰.咩事都冇發生只係覺得好攰好攰。'
    #data.loc[278,'suicide_means_or_in_committing_suicide'] = False    
    ###
    
    
    train = data
    train_text = train['helpseeker_msg'].to_list()
# build dict... fake index - is_exit
index_isexit = dict()
for i in range(data.shape[0]):
    index_isexit[i] = data.iloc[i]['is_exit']

# load word2vec 
cn_model = word2vec.KeyedVectors.load_word2vec_format('open_cn_word2vec_1010.model', binary=False)

import jieba
# jieba.set_dictionary('./hkcantonesedict.txt')
# jieba.load_userdict('./combined.txt')
jieba.set_dictionary('dict.txt')
jieba.load_userdict('hk_dict.txt')

#===========for counsellor==============
#===========for counsellor==============
train_text_counsellor = train['counsellor_msg'].to_list()

#%%
def tokenizer(train_text):
    train_tokens = []
    for i, text in enumerate(train_text):
        
        try:text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        except: text = '介手'

        cut = jieba.cut(text,cut_all = False)
        cut_list = [ i for i in cut ]
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.wv.vocab[word].index
            except KeyError:
                cut_list[i] = 0
        train_tokens.append(cut_list)
    return train_tokens

def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.wv.index2word[i]
        else:
            text = text + ' '
    return text

from tensorflow.keras import backend as K
'''
Compatible with tensorflow backend
'''

def focal_loss(gamma=2., alpha=.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

import pickle
def load_obj(name ):
    with open('../netvec/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)    
entity2vec = load_obj('entity2vec')
entities = list(entity2vec.keys())

#%%
# =================
train_tokens = tokenizer(train_text)
#def train_model(train_tokens, entities):
    # length of all tokens
num_tokens = [ len(tokens) for tokens in train_tokens ]
num_tokens = np.array(num_tokens)
print(num_tokens)
# average
# np.mean(num_tokens)

#max_tokens
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens
print(max_tokens)

embedding_dim = cn_model['返工'].shape[0]
print(embedding_dim)

num_words=len(cn_model.wv.vocab)

# load embedding_matrix for keras
embedding_matrix = np.zeros((num_words, embedding_dim))

# ----- substitute the word vec --------
## with KG embedding
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.wv.index2word[i]]
    if KG_embedding is True:
        if cn_model.wv.index2word[i] in entities: 
            embedding_matrix[i,:] = entity2vec[cn_model.wv.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')

# padding & truncating
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                            padding='pre', truncating='pre')
#print (train_pad)
# 0 to represent one that larger than num_words
train_pad[ train_pad>=num_words ] = 0

# corresponing to your label sequence
train_target = np.concatenate((np.ones(ones),np.zeros(zeros)))

#datavec-fake index
datavec_indx, datavec_label= dict(), dict()
cnt = 0
for row in train_pad:
    datavec_indx[tuple(row[-20:])] = cnt
    datavec_label[tuple(row[-20:])] = train_target[cnt]
    cnt += 1

# train size and test size
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.2,
                                                    random_state=randstate)

#===========for counsellor==============
#===========for counsellor==============
train_tokens_counsellor = tokenizer(train_text_counsellor)
num_tokens_counsellor = [ len(tokens) for tokens in train_tokens_counsellor ]
num_tokens_counsellor = np.array(num_tokens_counsellor)
print(num_tokens_counsellor)

#max_tokens
max_tokens_counsellor = np.mean(num_tokens_counsellor) + 2 * np.std(num_tokens_counsellor)
max_tokens_counsellor = int(max_tokens_counsellor)
max_tokens_counsellor
print(max_tokens_counsellor)

#embedding_dim/ num_words/ embedding_matrix/ train_target are general information (depending on wordvocab)

# padding & truncating
train_pad_counsellor = pad_sequences(train_tokens_counsellor, maxlen=max_tokens_counsellor,
                            padding='pre', truncating='pre')

# 0 to represent one that larger than num_words
train_pad_counsellor[ train_pad_counsellor>=num_words ] = 0

# train size and test size
X_train_counsellor, X_test_counsellor, y_train_useless, y_test_useless = train_test_split(train_pad_counsellor,
                                                    train_target,
                                                    test_size=0.2,
                                                    random_state=randstate)

#-----------keywords_indicator-----------
indicator = train['suicide_means_or_in_committing_suicide']
X_train_indicator, X_test_indicator, y_train_useless, y_test_useless = train_test_split(indicator,
                                                    train_target,
                                                    test_size=0.2,
                                                    random_state=randstate)
#%%
print ('model building ...') 
from tensorflow.keras.models import Model

emb = Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_tokens,trainable=False)

emb1 = Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_tokens,trainable=False)

emb2 = Embedding(num_words,embedding_dim,weights=[embedding_matrix],input_length=max_tokens_counsellor,trainable=False)

if multi_input:
    sequence_input1 = Input(shape=(max_tokens, ), dtype='int32')
    
    sequence_input2 = Input(shape=(max_tokens_counsellor, ), dtype='int32')

    keywords_indicator = Input(shape=(1, ), dtype='float32')
                    
    embedded_sequences1 = emb1(sequence_input1)
    
    embedded_sequences2 = emb2(sequence_input2)
    
    #counsellor_msg1 = Bidirectional(LSTM(units=32,return_sequences=False))(embedded_sequences1)
    counsellor_msg1 = Bidirectional(LSTM(units=16,return_sequences=True))(embedded_sequences1)
    counsellor_msg1 = LSTM(units=32,return_sequences=False)(counsellor_msg1)
    
    #counsellor_msg2 = Bidirectional(LSTM(units=32,return_sequences=False))(embedded_sequences2)
    counsellor_msg2 = Bidirectional(LSTM(units=16,return_sequences=True))(embedded_sequences2)
    counsellor_msg2 = LSTM(units=8,return_sequences=False)(counsellor_msg2)

    counsellor_msg = concatenate([counsellor_msg1, counsellor_msg2])      
    
    #reture_value = LSTM(units=16,return_sequences=False)(counsellor_msg)
    reture_value = Dense(units=8)(counsellor_msg)
    
    reture_value = concatenate([reture_value, keywords_indicator])  

    #reture_value = Dense(3, name = 'viz')(reture_value)
    
    drop = Dropout(0.2)(reture_value)
        
    out = Dense(1, activation='sigmoid', name = 'out')(drop)
    
    optimizer = Adam(lr=0.01)
    
    model = Model(inputs=[sequence_input1, sequence_input2, keywords_indicator],outputs=out)

else:
    '''
    sequence_input = Input(shape=(max_tokens, ), dtype='int32')
                        
    embedded_sequences = emb(sequence_input)
        
    counsellor_msg = Bidirectional(LSTM(units=32,return_sequences=True))(embedded_sequences)
            
    reture_value = LSTM(units=16,return_sequences=False)(counsellor_msg)
    #reture_value = Dense(units=8)(counsellor_msg)
    
    drop = Dropout(0.5)(reture_value)
    
    out = Dense(1, activation='sigmoid')(drop)
    
    optimizer = Adam(lr=0.001)
    
    model = Model(sequence_input, outputs=out)
    '''
    
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=False))
    
    model.add(Bidirectional(LSTM(units=32,return_sequences=True))) 
    model.add(LSTM(units=16,return_sequences=False)) 
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) 
    optimizer = Adam(lr=0.001)
    
print (model.summary())

model.compile(loss = [focal_loss(alpha=.75, gamma=2.)],
              optimizer=optimizer,
              metrics=['accuracy'])

#%%
# ---------MODEL SETTING------------
# checkpoint path set
path_checkpoint = 'openup_conversation.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_acc',
                                      verbose=1, save_weights_only=True,
                                      save_best_only=True)

if load_weights:
    try:
        print ('loading historical model...')
        model.load_weights(path_checkpoint)
    except Exception as e:
        print(e)

lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1, min_lr=1e-5, patience=0,
                                   verbose=1)

# early stopping
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# callbacks
callbacks = [ checkpoint,lr_reduction]

'''
# Output the Dx2vec embedding
from tensorflow.keras.models import Model
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('viz').output)
intermediate_output = intermediate_layer_model.predict([X_test,X_test_counsellor,X_test_indicator])
THREED_PATH = 'D:/research/openai/yucan/model/three_dim_visualization/'
np.savetxt(THREED_PATH+"chat2vec_emb.csv", intermediate_output, delimiter=",")
np.savetxt(THREED_PATH+"chat2vec_emb_class.csv", y_test)
'''
#%%
# ---------MODEL FITTING------------

if multi_input:
    
    history = model.fit([X_train,X_train_counsellor, X_train_indicator], y_train,epochs=1,batch_size=256,callbacks=callbacks,
                        validation_data=([X_test,X_test_counsellor,X_test_indicator], y_test))
    print(history.history.keys())
else:
    history = model.fit(X_train, y_train,epochs=3,batch_size=256,callbacks=callbacks,
                        validation_data=(X_test, y_test))
    print(history.history.keys())

model.save('binary_conversation.h5')

np.array([1] * X_train.shape[0]).shape
#%%
# ========model test==============
print ('model testing ...')


if multi_input: 
    
    y_val = model.predict([X_test,X_test_counsellor,X_test_indicator], verbose=0)
    
else:      
   
    y_val = model.predict(X_test, verbose=0)
    
y_value = deepcopy(y_val)
y_value[y_value>=0.5] = 1
y_value[y_value<0.5] = 0
print(classification_report(y_test, y_value))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_value))
#%%
indx_in_Xtest = 0
for row in X_test: 
    if datavec_label[tuple(row[-20:])] == 1:
        if datavec_indx[tuple(row[-20:])] == 278:
        #if True:
            print ('the index in the original text data (refer to df:data) is...', datavec_indx[tuple(row[-20:])])
            print ('the risk in y_test is...',y_val[indx_in_Xtest])
    indx_in_Xtest+=1
#data.to_csv('check_risk_msg.csv', encoding= 'utf-8-sig')    
#%%
    '''
isexit = []

for row in X_test:   
    
    isexit.append(index_isexit[datavec_indx[tuple(row[-20:])]])

len(isexit)
tmp = [i for i in isexit if i]
len(tmp)     

exit_label, exit_groundtruth, not_exit_label,not_exit_groundtruth = [], [], [], []

for i, j, k in zip(isexit, y_value, y_test):
    
    if i:        
        exit_label.append(j)
        exit_groundtruth.append(k)
    
    else: 
        not_exit_label.append(j)        
        not_exit_groundtruth.append(k)

print('exit risk prediction...')
print(classification_report(exit_groundtruth, exit_label,digits=3))
print('on-going risk prediction...')
print(classification_report(not_exit_groundtruth, not_exit_label,digits=3))
'''
# p-r/roc curve
'''
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, roc_auc_score      
def plot_pr(auc_score, precision, recall, label=None, pr=True):  
    pylab.figure(num=None, figsize=(6, 5))  
    pylab.xlim([0.0, 1.0])  
    pylab.ylim([0.0, 1.0])  
     
    if pr:
        pylab.xlabel('Recall') 
        pylab.ylabel('Precision')
        pylab.title('P/R CURVE') 
    else:
        pylab.xlabel('FPR') 
        pylab.ylabel('TPR')
        pylab.title('ROC CURVE') 
     
    pylab.fill_between(recall, precision, alpha=0.5)  
    pylab.grid(True, linestyle='-', color='0.75')  
    pylab.plot(recall, precision, lw=1)      
    pylab.show()
precision, recall, thresholds = precision_recall_curve(y_test, y_val)
fpr, tpr, thresholds = roc_curve(y_test, y_val, pos_label=1)
plot_pr(0.5,  precision,recall, "pos")
plot_pr(0.5,  tpr,fpr, "pos",pr=False)
'''




   
