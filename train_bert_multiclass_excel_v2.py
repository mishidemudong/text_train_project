#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:46:32 2022

@author: liang
"""


import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import random
import numpy as np
import keras.backend as K
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense, Concatenate, Layer, Conv1D, GlobalMaxPool1D, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report
import datetime

set_gelu('tanh')  # 切换gelu版本
today = datetime.date.today().strftime('%m_%d') + 'tiny'
config = {}
config['maxlen'] = 256
config['batch_size'] = 32
config['pretrain_type'] = 'bert'
config['config_path'] = '/media/liang/Nas/PreTrainModel/bert/eng_tiny_bert/uncased_L-6_H-128_A-2/bert_config.json'
config['checkpoint_path'] = '/media/liang/Nas/PreTrainModel/bert/eng_tiny_bert/uncased_L-6_H-128_A-2/bert_model.ckpt'
config['dict_path'] = '/media/liang/Nas/PreTrainModel/bert/eng_tiny_bert/uncased_L-6_H-128_A-2/vocab.txt'
model_savepath = f'./model/checkpoint/tiny_bert_xiuzheng_quchong_{today}.weights'

# config['config_path'] = '/media/liang/Nas/PreTrainModel/bert/eng_bert/bert-base-uncased/bert_config.json'
# config['checkpoint_path'] = '/media/liang/Nas/PreTrainModel/bert/eng_bert/bert-base-uncased/bert_model.ckpt'
# config['dict_path'] = '/media/liang/Nas/PreTrainModel/bert/eng_bert/bert-base-uncased/vocab.txt'
# model_savepath = './model/modelcheckpoint/bert_xiuzheng_quchong_0802.weights'

def load_data(filepath,sheet_names):
    """加载数据
    单条格式：(文本, 标签id)
    """
    oridf = pd.DataFrame()
    for sheet in tqdm(sheet_names):
        if sheet != '层级结构':
            df = pd.read_excel(filepath, sheet_name=sheet).drop_duplicates('id').dropna(axis=0).drop(['name','label'],axis=1)
            df['real_name'] = [sheet] * df.shape[0]
            oridf = oridf.append(df)
            # for index, item in tqdm(df.iterrows()):
            #     # print(item)
            #     D.append((item['body'], sheet))
        
    return oridf


# 加载数据集
filepath = './datasets/pred_0803_ori_quchong_correct_version2.xlsx' #pred_07_26_ori_quchong_已确认,#pred_0802_ori_quchong_correct,pred_0803_ori_quchong_correct_version2
# excel_reader=pd.ExcelFile(filepath) # 指定文件 
# sheet_names = excel_reader.sheet_names
oridf = pd.read_excel(filepath).drop_duplicates('id')[['id','chatter_id','body','real_name']]#.drop(['pred_name','pred_label','pred_proba'],axis=1)


otherdf = pd.read_csv('/media/liang/NLPProject/NLP/text_clc/content-moderation-project/test.csv')

otherdata = []
for index, item in tqdm(otherdf.iterrows()):
    if len(item['comment_text']) < 128 and len(item['comment_text'].replace(' ','')) > 5:
        otherdata.append((item['comment_text'].strip().replace('=',''),'其他问题'))
        
otherdata_sample = random.sample(otherdata, 500)
otherdata_sample.extend([('Thank you!', '其他问题'),('You too!', '其他问题'),('It is a test', '其他问题'),('I just did', '其他问题')])

labelfilepath = './datasets/Answer bot数据分类整理-0726-修正.xlsx'
excel_reader=pd.ExcelFile(labelfilepath) # 指定文件 
label_names = excel_reader.sheet_names

label_dict ={}
label = 0
for name in label_names:
    if name != '层级结构':
        label_dict[name] = label
        label += 1
label_dict['其他问题'] = 37
# str_length = {}
# for item  in df['data_str'].tolist():
#     str_length[str(len(item))] = str_length.get(str(len(item)), 0) + 1

for sheet in ['更换账户','被盗类case','其他法务类case', '充提开关', '提现到错误链路(类似KCC这种)&提现到错误地址', 'APP&网页问题', '手机绑定解绑', '杠杆问题', '下币咨询和提现', '费用相关']:#'APP&网页问题',
    df = pd.read_excel(labelfilepath, sheet_name=sheet).drop_duplicates('id').dropna(axis=0).drop(['name','label'],axis=1)
    df['real_name'] = [sheet] * df.shape[0]
    oridf = oridf.append(df)


config['label2id'] = label_dict
config['id2label'] = {v:k for k,v in config['label2id'].items()}
config['num_classes'] = len(config['label2id'])

data = [(item['body'], item['real_name']) for index, item in tqdm(oridf.iterrows())]

#add other class
data += otherdata_sample 

savedata = pd.DataFrame(data)
savedata.to_csv(f"./data/train_xiuzheng_{today}.csv", index = False)


staticdf = pd.DataFrame(savedata[[1]].value_counts())
# staticdf = pd.DataFrame(oridf[['real_name']].value_counts())

staticdf.to_excel(f'./result/label_static_{today}.xlsx')

# 建立分词器
tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label_dict[label]])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def buildmodel(config):
    
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config['config_path'],
        checkpoint_path=config['checkpoint_path'],
        model=config['pretrain_type'],
        return_keras_model=False,
    )
    
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = Dropout(0.1)(output)
    output = Dense(
        units=config['num_classes'],
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)
    
    model = keras.models.Model(bert.model.input, output)
    model.summary()
    
    return model


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(model_savepath)
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    
    # 转换数据集
    train_data, test_data= train_test_split(data, test_size=0.3, random_state=2022)
    train_data, valid_data= train_test_split(train_data, test_size=0.15, random_state=2022)
    
    train_generator = data_generator(train_data, config['batch_size'])
    valid_generator = data_generator(valid_data, config['batch_size'])
    test_generator = data_generator(test_data, config['batch_size'])

    evaluator = Evaluator()
    
    model = buildmodel(config)
    # 派生为带分段线性学习率的优化器。
    # 其中name参数可选，但最好填入，以区分不同的派生优化器。
    AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        # optimizer=Adam(1e-5),  # 用足够小的学习率
        optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
            1000: 1,
            2000: 0.1
        }),
        metrics=['accuracy'],
    )
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

    model.load_weights(model_savepath)
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))
    
    real_label = []
    pred_label = []
    for item in tqdm(test_data):
        token_ids, segment_ids = tokenizer.encode(item[0], maxlen=config['maxlen'])
        pred = model.predict([[token_ids], [segment_ids]]).argmax(axis=1)[0]
        pred_label.append(pred)
        real_label.append(label_dict[item[1]])


    print(classification_report(real_label, pred_label, target_names=list(label_dict.keys())))
    result = classification_report(real_label, pred_label, target_names=list(label_dict.keys()), output_dict=True)
    
    res_df = pd.DataFrame(result).stack().unstack(0)
    res_df.to_excel(f'./result/report_bert_ori_{today}.xlsx')
    print(classification_report(real_label, pred_label))
    
    
    config['config_path'] = './model/checkpoint/bert_config.json'
    config['checkpoint_path'] = './model/checkpoint/bert_model.ckpt'
    config['dict_path'] = './model/checkpoint/vocab.txt'

    config['model_checkpoint'] = "./model/checkpoint/best_model.weights"
    
    import json
    config_savepath = './intent_model.json'
    output = open(config_savepath, 'w', encoding='utf-8')
    json.dump(config, output,ensure_ascii=False)
    
    output.close()
    
    
    from sklearn.metrics import confusion_matrix
    from pprint import pprint
    pprint(confusion_matrix([config['id2label'][item] for item in real_label], [config['id2label'][item] for item in pred_label], labels = list(label_dict.keys())))
    
    a = confusion_matrix([config['id2label'][item] for item in real_label], [config['id2label'][item] for item in pred_label], labels = list(label_dict.keys()))
    
    '''
******************************************************************************************************
	                    precision	recall	f1-score	support
谷歌问题	0.894736842	0.96835443	0.930091185	158
手机绑定解绑	0.853448276	0.804878049	0.828451883	123
导出交易记录	0.87012987	0.881578947	0.875816993	76
短信邮件code	0.889784946	0.940340909	0.914364641	352
更换账户	0.890625	0.721518987	0.797202797	79
删除账户	0.931818182	0.759259259	0.836734694	54
资产账户相关	0.978723404	0.686567164	0.807017544	67
API	0.95	0.974358974	0.962025316	39
P2P申诉	0.987209302	0.984918794	0.986062718	862
SEPA入金	0.970833333	0.966804979	0.968814969	241
Capitual入金	1	0.968	0.983739837	125
银行卡买币	0.973770492	0.976973684	0.975369458	304
Advcash出入金咨询	1	0.958333333	0.978723404	24
第三方买币	0.96969697	0.914285714	0.941176471	35
除上面几种具体渠道外的法币入金	0.84	0.84	0.84	25
充值	0.899736148	0.949860724	0.924119241	359
提现未到账	0.913043478	0.976744186	0.943820225	215
提现到错误链路(类似KCC这种)&提现到错误地址	0.988235294	1	0.99408284	84
充提开关	0.870967742	0.818181818	0.84375	99
其他提现问题	0.946808511	0.89	0.917525773	100
现货相关	0.913043478	0.823529412	0.865979381	51
费用相关	0.913907285	0.965034965	0.93877551	143
合约	0.905555556	0.898071625	0.901798064	363
KYC问题	0.830188679	0.897959184	0.862745098	49
杠杆问题	0.828418231	0.950769231	0.885386819	325
杠杆借贷	0.925	0.804347826	0.860465116	46
交易机器人	0.939393939	0.815789474	0.873239437	38
杠杆代币	0.918367347	0.918367347	0.918367347	49
平台活动	0.875706215	0.922619048	0.898550725	168
被盗类case	0.859259259	0.943089431	0.899224806	123
其他法务类case	0.914529915	0.938596491	0.926406926	114
KuCoin Earn	0.861788618	0.890756303	0.876033058	119
APP&网页问题	0.805555556	0.725	0.763157895	80
下币咨询和提现	0.871794872	0.708333333	0.781609195	48
注册登录	0.96037296	0.951501155	0.955916473	433
交易密码	0.924369748	0.916666667	0.920502092	240
合约强平	0.9625	0.675438596	0.793814433	114
其他问题	0.944	0.874074074	0.907692308	135
accuracy	0.9222644	0.9222644	0.9222644	0.9222644
macro avg	0.915087354	0.884234319	0.89680407	6059
weighted avg	0.923751949	0.9222644	0.921320236	6059

    '''
    

else:
    pred_data = oridf
    
    def choose(data):
        body = []
        id_ = []
        chatid = []
        real_name = []
        for index, item in tqdm(data.iterrows()):
            # print(item)
            if not isinstance(item['body'], str):
                continue
            else:
                body.append(item['body'])
                id_.append(item['id'])
                chatid.append(item['chatter_id'])
                real_name.append(item['real_name'])
    
        return body, real_name, id_, chatid
    
    body, real_name, id_, chatid = choose(pred_data)
    xpred_label = []
    xpred_proba = []
    for item in tqdm(body):
        token_ids, segment_ids = tokenizer.encode(item, maxlen=config['maxlen'])

        y_proba = model.predict([[token_ids], [segment_ids]])
        y_pred = y_proba.argmax(axis=1)[0]
        
        xpred_label.append(y_pred)
        xpred_proba.append(max(y_proba[0]))
        
        
    result = pd.DataFrame()
    result['id'] = id_
    result['chatter_id'] = chatid
    result['body'] = body
    result['real_name'] = real_name
    result['pred_name'] = [config['id2label'][item] for item in xpred_label]
    result['pred_label'] = xpred_label
    result['pred_proba'] = xpred_proba
    
    result.to_excel(f'./result/pred_{today}.xlsx')
    
    
    ###########################################################################################################

    model = buildmodel(config)

    model.load_weights(model_savepath)
    
    pred_data = pd.read_excel('./result/pred_07_26_new_delta.xlsx').drop_duplicates(['id'],keep=False).drop(['pred_name','pred_label','pred_proba'],axis=1)
    
    
    def choose(data):
        body = []
        id_ = []
        chatid = []
        for index, item in tqdm(data.iterrows()):
            # print(item)
            if not isinstance(item['body'], str):
                continue
            else:
                body.append(item['body'])
                id_.append(item['id'])
                chatid.append(item['chatter_id'])
    
        return body, id_, chatid
    
    body, id_, chatid = choose(pred_data)
    

    xpred_label = []
    xpred_proba = []
    for item in tqdm(body):
        token_ids, segment_ids = tokenizer.encode(item, maxlen=config['maxlen'])

        y_proba = model.predict([[token_ids], [segment_ids]])
        y_pred = y_proba.argmax(axis=1)[0]
        
        xpred_label.append(y_pred)
        xpred_proba.append(max(y_proba[0]))
        
        
    result = pd.DataFrame()
    result['id'] = id_
    result['chatter_id'] = chatid
    result['body'] = body
    result['pred_name'] = [config['id2label'][item] for item in xpred_label]
    result['pred_label'] = xpred_label
    result['pred_proba'] = xpred_proba
    
    result.to_excel(f'./result/pred_{today}_new_delta.xlsx')
    
    
    
        
    
