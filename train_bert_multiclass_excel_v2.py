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

# filepath2 = './result/131条数据校准.xlsx'
# xiuzhengdf = pd.read_excel(filepath2).drop(['pred_name','pred_label','pred_proba'],axis=1)

# # oridf2 = oridf.set_index('id').combine_first(xiuzhengdf.set_index('id')).reset_index().drop_duplicates('id', keep='first')

# oridf.update(xiuzhengdf)

# oridf = oridf.drop_duplicates('id', keep='first')
# oridf.to_excel('./datasets/pred_0802_ori_quchong_correct.xlsx')
# oridf2 = pd.merge(oridf, xiuzhengdf, how='left', on=['id','chatter_id','body'])
    #             id_.append(item['id'])
    #             chatid.append(item['chatter_id']])

# 加载数据集
# filepath = './datasets/ADA二批次数据校准_0803_drop.xlsx'
# df1 = pd.read_excel(filepath, sheet_name='谷歌问题').drop_duplicates('id').drop(['pred_name','pred_label','pred_proba'],axis=1)
# df1 = df1.drop_duplicates('body')[['id','chatter_id','body']]
# df1['real_name'] = ['谷歌问题']*df1.shape[0]
# df2 = pd.read_excel(filepath, sheet_name='充值').drop_duplicates('id').drop(['pred_name','pred_label','pred_proba'],axis=1)
# df2 = df2.drop_duplicates('body')[['id','chatter_id','body']]
# df2['real_name'] = ['充值']*df2.shape[0]


# df3 = pd.read_excel(filepath, sheet_name='下币咨询和提现').drop_duplicates('id').drop(['pred_name','pred_label','pred_proba'],axis=1)
# df3 = df3.drop_duplicates('body')[['id','chatter_id','body']]
# df3['real_name'] = ['下币咨询和提现']*df3.shape[0]

# df4 = pd.read_excel(filepath, sheet_name='KuCoin Earn').drop_duplicates('id').drop(['pred_name','pred_label','pred_proba'],axis=1)
# df4 = df4.drop_duplicates('body')[['id','chatter_id','body']]
# df4['real_name'] = ['KuCoin Earn']*df4.shape[0]


# oridf = oridf.append(df1)
# oridf = oridf.append(df2)
# oridf = oridf.append(df3)
# oridf = oridf.append(df4)
# oridf.to_excel('./datasets/pred_0803_ori_quchong_correct_version2.xlsx')

# data = label_df.replace(xiuzheng_df)

# xiuzheng_df2 = label_df[label_df['real_name']=='合约强平']
# oridf = oridf.replace(xiuzheng_df)
# oridf = oridf.replace(xiuzheng_df2)


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


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x


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


def buildmodelbertcnn(config):
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config['config_path'],
        checkpoint_path=config['checkpoint_path'],
        model=config['pretrain_type'],
        return_keras_model=False,
    )
    
    #####
    num_hidden_layers = [0,1,3,5]
    features_layers = [bert.model.get_layer('Transformer-%d-FeedForward-Norm' % i).output \
                                            for i in num_hidden_layers]
    outputs = Concatenate()(features_layers)
    outputs = NonMaskingLayer()(outputs)

    ####CNN define
    output_cnn = []
    cnn_sizes = [2 ,3, 4]
    
    for size in cnn_sizes:
        output = Conv1D(filters=K.int_shape(outputs)[-1], kernel_size = [size], 
                        strides = 1, padding='same', activation='relu'
                        )(outputs)
        
        output = GlobalMaxPool1D()(output)
        
        output_cnn.append(output)
        
    cnn_output = Concatenate()(output_cnn)
    cnn_output = Dropout(0.2)(cnn_output)

    output = Dense(
        units=config['num_classes'],
        activation='softmax',
        kernel_initializer=bert.initializer
    )(cnn_output)

    model = keras.models.Model(bert.model.input, output)
    model.summary()
    
    return model

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


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
    # model = buildmodelbertcnn(config)
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
    eng_tiny
                                   

    bert base
    precision    recall  f1-score   support

                    谷歌问题       0.82      0.59      0.69        39
                  手机绑定解绑       0.71      0.82      0.76        33
                    交易密码       0.62      0.99      0.76       334
                  导出交易记录       0.91      0.90      0.91        59
                    注册登录       0.87      0.71      0.78       586
                短信邮件code       0.89      0.76      0.82       336
                    更换账户       1.00      0.70      0.83        37
                    删除账户       0.94      0.97      0.96        35
                  资产账户相关       0.78      0.86      0.82        29
                     API       0.93      1.00      0.96        38
                   P2P申诉       0.99      0.99      0.99       811
                  SEPA入金       0.94      0.95      0.95       210
              Capitual入金       0.98      1.00      0.99        82
                   银行卡买币       0.95      0.96      0.95       239
            Advcash出入金咨询       0.78      1.00      0.88        18
                   第三方买币       0.97      0.90      0.94        41
         除上面几种具体渠道外的法币入金       0.74      0.61      0.67        23
                      充值       0.99      0.96      0.97       218
                   提现未到账       0.97      0.95      0.96       187
提现到错误链路(类似KCC这种)&提现到错误地址       1.00      0.98      0.99        42
                    充提开关       0.67      0.88      0.76        41
                  其他提现问题       0.93      0.90      0.91        83
                    现货相关       0.92      0.89      0.90        37
                    费用相关       0.97      0.79      0.87        43
                      合约       0.89      0.76      0.82       314
                    合约强平       0.59      0.83      0.69       164
                   KYC问题       0.98      0.92      0.95        53
                    杠杆问题       0.70      0.61      0.65       153
                    杠杆借贷       0.85      0.85      0.85        53
                   交易机器人       0.87      0.85      0.86        40
                    杠杆代币       1.00      0.98      0.99        44
                    平台活动       0.97      0.97      0.97       354
                 被盗类case       1.00      0.82      0.90        45
               其他法务类case       0.96      0.84      0.90        51
             KuCoin Earn       0.95      0.93      0.94        58
                APP&网页问题       0.74      0.77      0.76        22
                 下币咨询和提现       0.94      0.79      0.86        38

                accuracy                           0.88      4990
               macro avg       0.88      0.86      0.87      4990
            weighted avg       0.89      0.88      0.88      4990


              precision    recall  f1-score   support

           0       0.82      0.59      0.69        39
           1       0.71      0.82      0.76        33
           2       0.62      0.99      0.76       334
           3       0.91      0.90      0.91        59
           4       0.87      0.71      0.78       586
           5       0.89      0.76      0.82       336
           6       1.00      0.70      0.83        37
           7       0.94      0.97      0.96        35
           8       0.78      0.86      0.82        29
           9       0.93      1.00      0.96        38
          10       0.99      0.99      0.99       811
          11       0.94      0.95      0.95       210
          12       0.98      1.00      0.99        82
          13       0.95      0.96      0.95       239
          14       0.78      1.00      0.88        18
          15       0.97      0.90      0.94        41
          16       0.74      0.61      0.67        23
          17       0.99      0.96      0.97       218
          18       0.97      0.95      0.96       187
          19       1.00      0.98      0.99        42
          20       0.67      0.88      0.76        41
          21       0.93      0.90      0.91        83
          22       0.92      0.89      0.90        37
          23       0.97      0.79      0.87        43
          24       0.89      0.76      0.82       314
          25       0.59      0.83      0.69       164
          26       0.98      0.92      0.95        53
          27       0.70      0.61      0.65       153
          28       0.85      0.85      0.85        53
          29       0.87      0.85      0.86        40
          30       1.00      0.98      0.99        44
          31       0.97      0.97      0.97       354
          32       1.00      0.82      0.90        45
          33       0.96      0.84      0.90        51
          34       0.95      0.93      0.94        58
          35       0.74      0.77      0.76        22
          36       0.94      0.79      0.86        38

    accuracy                           0.88      4990
   macro avg       0.88      0.86      0.87      4990
weighted avg       0.89      0.88      0.88      4990

******************************************************************************************************

                        P2P申诉                        2866
                        注册登录                        1409
                        合约                          1270
                        短信邮件code                    1170
                        银行卡买币                        943
                        SEPA入金                       819
                        交易密码                         795
                        提现未到账                        711
                        平台活动                         582
                        杠杆问题                         539
                        充值                           436
                        Capitual入金                   361
                        合约强平                         343
                        其他提现问题                       305
                        导出交易记录                       245
                        谷歌问题                         207
                        KYC问题                        200
                        KuCoin Earn                  197
                        现货相关                         194
                        其他法务类case                    191
                        费用相关                         185
                        被盗类case                      177
                        手机绑定解绑                       175
                        杠杆代币                         171
                        杠杆借贷                         162
                        资产账户相关                       160
                        充提开关                         160
                        删除账户                         157
                        交易机器人                        153
                        更换账户                         144
                        提现到错误链路(类似KCC这种)&提现到错误地址     141
                        API                          140
                        第三方买币                        136
                        APP&网页问题                     118
                        除上面几种具体渠道外的法币入金              104
                        Advcash出入金咨询                  84
                        下币咨询和提现                       63

                                            precision    recall  f1-score   support
        
                            谷歌问题       0.67      0.88      0.76        42
                          手机绑定解绑       0.63      0.68      0.66        28
                          导出交易记录       0.92      0.96      0.94        49
                        短信邮件code       0.93      0.93      0.93       237
                            更换账户       0.90      0.63      0.75        30
                            删除账户       0.85      0.88      0.86        40
                          资产账户相关       0.90      0.76      0.83        34
                             API       0.95      0.83      0.89        24
                           P2P申诉       0.99      0.99      0.99       541
                          SEPA入金       0.99      0.97      0.98       173
                      Capitual入金       1.00      1.00      1.00        66
                           银行卡买币       0.94      0.98      0.96       193
                    Advcash出入金咨询       1.00      1.00      1.00        17
                           第三方买币       0.88      1.00      0.93        14
                 除上面几种具体渠道外的法币入金       0.93      0.93      0.93        15
                              充值       0.85      0.91      0.88        87
                           提现未到账       0.93      0.98      0.95       130
        提现到错误链路(类似KCC这种)&提现到错误地址       1.00      0.97      0.99        35
                            充提开关       0.77      0.53      0.63        32
                          其他提现问题       0.92      0.91      0.92        67
                            现货相关       0.97      0.85      0.91        34
                            费用相关       0.90      0.90      0.90        42
                              合约       0.94      0.95      0.95       257
                           KYC问题       0.91      0.91      0.91        34
                            杠杆问题       0.82      0.92      0.87        99
                            杠杆借贷       0.88      0.81      0.85        37
                           交易机器人       0.79      0.96      0.87        28
                            杠杆代币       0.98      0.98      0.98        42
                            平台活动       0.93      0.95      0.94       124
                         被盗类case       0.93      0.95      0.94        44
                       其他法务类case       0.91      0.91      0.91        35
                     KuCoin Earn       0.77      0.92      0.84        36
                        APP&网页问题       0.93      0.46      0.62        28
                         下币咨询和提现       0.00      0.00      0.00        10
                            注册登录       0.96      0.96      0.96       298
                            交易密码       0.97      0.91      0.94       169
                            合约强平       0.82      0.82      0.82        72
        
                        accuracy                           0.93      3243
                       macro avg       0.88      0.86      0.86      3243
                    weighted avg       0.93      0.93      0.93      3243
        
                      precision    recall  f1-score   support
        
                   0       0.67      0.88      0.76        42
                   1       0.63      0.68      0.66        28
                   2       0.92      0.96      0.94        49
                   3       0.93      0.93      0.93       237
                   4       0.90      0.63      0.75        30
                   5       0.85      0.88      0.86        40
                   6       0.90      0.76      0.83        34
                   7       0.95      0.83      0.89        24
                   8       0.99      0.99      0.99       541
                   9       0.99      0.97      0.98       173
                  10       1.00      1.00      1.00        66
                  11       0.94      0.98      0.96       193
                  12       1.00      1.00      1.00        17
                  13       0.88      1.00      0.93        14
                  14       0.93      0.93      0.93        15
                  15       0.85      0.91      0.88        87
                  16       0.93      0.98      0.95       130
                  17       1.00      0.97      0.99        35
                  18       0.77      0.53      0.63        32
                  19       0.92      0.91      0.92        67
                  20       0.97      0.85      0.91        34
                  21       0.90      0.90      0.90        42
                  22       0.94      0.95      0.95       257
                  23       0.91      0.91      0.91        34
                  24       0.82      0.92      0.87        99
                  25       0.88      0.81      0.85        37
                  26       0.79      0.96      0.87        28
                  27       0.98      0.98      0.98        42
                  28       0.93      0.95      0.94       124
                  29       0.93      0.95      0.94        44
                  30       0.91      0.91      0.91        35
                  31       0.77      0.92      0.84        36
                  32       0.93      0.46      0.62        28
                  33       0.00      0.00      0.00        10
                  34       0.96      0.96      0.96       298
                  35       0.97      0.91      0.94       169
                  36       0.82      0.82      0.82        72
        
            accuracy                           0.93      3243
           macro avg       0.88      0.86      0.86      3243
        weighted avg       0.93      0.93      0.93      3243
        
        
                                precision    recall  f1-score   support

                    谷歌问题       0.72      0.88      0.79        57
                  手机绑定解绑       0.79      0.65      0.71        46
                  导出交易记录       0.91      0.93      0.92        69
                短信邮件code       0.91      0.95      0.93       360
                    更换账户       0.97      0.70      0.81        47
                    删除账户       0.91      0.90      0.91        59
                  资产账户相关       0.88      0.67      0.76        45
                     API       0.89      0.87      0.88        38
                   P2P申诉       1.00      1.00      1.00       819
                  SEPA入金       0.97      0.96      0.97       265
              Capitual入金       0.97      1.00      0.99       108
                   银行卡买币       0.96      0.99      0.97       285
            Advcash出入金咨询       0.93      1.00      0.96        26
                   第三方买币       0.84      0.93      0.89        29
         除上面几种具体渠道外的法币入金       0.96      0.96      0.96        28
                      充值       0.88      0.91      0.89       133
                   提现未到账       0.92      0.96      0.94       191
提现到错误链路(类似KCC这种)&提现到错误地址       0.94      0.98      0.96        48
                    充提开关       0.72      0.60      0.65        52
                  其他提现问题       0.91      0.89      0.90        92
                    现货相关       0.94      0.94      0.94        53
                    费用相关       0.93      0.94      0.93        67
                      合约       0.92      0.98      0.95       368
                   KYC问题       0.95      0.95      0.95        63
                    杠杆问题       0.85      0.87      0.86       159
                    杠杆借贷       0.89      0.81      0.85        52
                   交易机器人       0.93      0.86      0.89        43
                    杠杆代币       0.93      0.92      0.92        59
                    平台活动       0.97      0.90      0.93       182
                 被盗类case       0.90      0.92      0.91        60
               其他法务类case       0.95      0.92      0.93        60
             KuCoin Earn       0.76      0.88      0.82        59
                APP&网页问题       0.83      0.68      0.75        37
                 下币咨询和提现       1.00      0.44      0.61        16
                    注册登录       0.96      0.96      0.96       434
                    交易密码       0.95      0.91      0.93       243
                    合约强平       0.85      0.81      0.83       112

                accuracy                           0.93      4864
               macro avg       0.91      0.88      0.89      4864
            weighted avg       0.93      0.93      0.93      4864

              precision    recall  f1-score   support

           0       0.72      0.88      0.79        57
           1       0.79      0.65      0.71        46
           2       0.91      0.93      0.92        69
           3       0.91      0.95      0.93       360
           4       0.97      0.70      0.81        47
           5       0.91      0.90      0.91        59
           6       0.88      0.67      0.76        45
           7       0.89      0.87      0.88        38
           8       1.00      1.00      1.00       819
           9       0.97      0.96      0.97       265
          10       0.97      1.00      0.99       108
          11       0.96      0.99      0.97       285
          12       0.93      1.00      0.96        26
          13       0.84      0.93      0.89        29
          14       0.96      0.96      0.96        28
          15       0.88      0.91      0.89       133
          16       0.92      0.96      0.94       191
          17       0.94      0.98      0.96        48
          18       0.72      0.60      0.65        52
          19       0.91      0.89      0.90        92
          20       0.94      0.94      0.94        53
          21       0.93      0.94      0.93        67
          22       0.92      0.98      0.95       368
          23       0.95      0.95      0.95        63
          24       0.85      0.87      0.86       159
          25       0.89      0.81      0.85        52
          26       0.93      0.86      0.89        43
          27       0.93      0.92      0.92        59
          28       0.97      0.90      0.93       182
          29       0.90      0.92      0.91        60
          30       0.95      0.92      0.93        60
          31       0.76      0.88      0.82        59
          32       0.83      0.68      0.75        37
          33       1.00      0.44      0.61        16
          34       0.96      0.96      0.96       434
          35       0.95      0.91      0.93       243
          36       0.85      0.81      0.83       112

    accuracy                           0.93      4864
   macro avg       0.91      0.88      0.89      4864
weighted avg       0.93      0.93      0.93      4864


                precision    recall  f1-score   support

                    谷歌问题       0.72      0.92      0.81        25
                  手机绑定解绑       0.92      0.76      0.83        59
                  导出交易记录       0.83      0.90      0.86        71
                短信邮件code       0.93      0.94      0.94       350
                    更换账户       0.96      0.74      0.84        35
                    删除账户       0.96      0.85      0.90        59
                  资产账户相关       0.81      0.78      0.80        55
                     API       0.88      0.97      0.92        37
                   P2P申诉       0.99      0.99      0.99       809
                  SEPA入金       0.96      0.98      0.97       256
              Capitual入金       0.97      1.00      0.99       106
                   银行卡买币       0.96      0.98      0.97       278
            Advcash出入金咨询       1.00      0.88      0.94        26
                   第三方买币       0.94      0.85      0.89        39
         除上面几种具体渠道外的法币入金       1.00      0.85      0.92        33
                      充值       0.79      0.87      0.83       145
                   提现未到账       0.92      0.94      0.93       195
提现到错误链路(类似KCC这种)&提现到错误地址       0.86      0.97      0.91        37
                    充提开关       0.87      0.45      0.59        60
                  其他提现问题       0.88      0.95      0.92        87
                    现货相关       0.97      0.95      0.96        62
                    费用相关       0.87      0.90      0.89        61
                      合约       0.94      0.94      0.94       393
                   KYC问题       0.94      0.89      0.91        65
                    杠杆问题       0.83      0.89      0.86       160
                    杠杆借贷       0.86      0.84      0.85        51
                   交易机器人       0.90      0.77      0.83        48
                    杠杆代币       0.94      0.98      0.96        49
                    平台活动       0.92      0.95      0.93       184
                 被盗类case       0.87      0.94      0.90        49
               其他法务类case       0.92      0.95      0.94        61
             KuCoin Earn       0.81      0.85      0.83        61
                APP&网页问题       0.83      0.69      0.75        29
                 下币咨询和提现       1.00      0.54      0.70        13
                    注册登录       0.97      0.96      0.96       444
                    交易密码       0.92      0.96      0.94       230
                    合约强平       0.84      0.82      0.83       103

                accuracy                           0.93      4825
               macro avg       0.91      0.88      0.88      4825
            weighted avg       0.93      0.93      0.93      4825


    bert base
                                precision    recall  f1-score   support
                    谷歌问题       0.79      0.77      0.78        57
                  手机绑定解绑       0.94      0.65      0.77        46
                  导出交易记录       0.97      1.00      0.99        69
                短信邮件code       0.90      0.96      0.93       360
                    更换账户       0.91      0.85      0.88        47
                    删除账户       0.95      0.97      0.96        59
                  资产账户相关       0.97      0.84      0.90        45
                     API       1.00      0.92      0.96        38
                   P2P申诉       0.99      1.00      0.99       819
                  SEPA入金       0.98      0.97      0.98       265
              Capitual入金       1.00      0.99      1.00       108
                   银行卡买币       0.95      1.00      0.97       285
            Advcash出入金咨询       0.93      1.00      0.96        26
                   第三方买币       0.90      0.97      0.93        29
         除上面几种具体渠道外的法币入金       0.97      1.00      0.98        28
                      充值       0.97      0.97      0.97       133
                   提现未到账       0.97      0.96      0.97       191
提现到错误链路(类似KCC这种)&提现到错误地址       0.98      1.00      0.99        48
                    充提开关       0.85      0.85      0.85        52
                  其他提现问题       0.98      0.97      0.97        92
                    现货相关       0.95      0.98      0.96        53
                    费用相关       0.92      0.91      0.92        67
                      合约       0.94      0.96      0.95       368
                   KYC问题       0.95      0.95      0.95        63
                    杠杆问题       0.87      0.87      0.87       159
                    杠杆借贷       0.89      0.90      0.90        52
                   交易机器人       0.89      0.93      0.91        43
                    杠杆代币       0.97      0.97      0.97        59
                    平台活动       0.97      0.95      0.96       182
                 被盗类case       0.90      0.93      0.92        60
               其他法务类case       0.97      0.95      0.96        60
             KuCoin Earn       0.95      0.92      0.93        59
                APP&网页问题       0.89      0.89      0.89        37
                 下币咨询和提现       0.88      0.88      0.88        16
                    注册登录       0.97      0.95      0.96       434
                    交易密码       0.97      0.91      0.94       243
                    合约强平       0.83      0.84      0.84       112

                accuracy                           0.95      4864
               macro avg       0.93      0.93      0.93      4864
            weighted avg       0.95      0.95      0.95      4864

              precision    recall  f1-score   support

           0       0.79      0.77      0.78        57
           1       0.94      0.65      0.77        46
           2       0.97      1.00      0.99        69
           3       0.90      0.96      0.93       360
           4       0.91      0.85      0.88        47
           5       0.95      0.97      0.96        59
           6       0.97      0.84      0.90        45
           7       1.00      0.92      0.96        38
           8       0.99      1.00      0.99       819
           9       0.98      0.97      0.98       265
          10       1.00      0.99      1.00       108
          11       0.95      1.00      0.97       285
          12       0.93      1.00      0.96        26
          13       0.90      0.97      0.93        29
          14       0.97      1.00      0.98        28
          15       0.97      0.97      0.97       133
          16       0.97      0.96      0.97       191
          17       0.98      1.00      0.99        48
          18       0.85      0.85      0.85        52
          19       0.98      0.97      0.97        92
          20       0.95      0.98      0.96        53
          21       0.92      0.91      0.92        67
          22       0.94      0.96      0.95       368
          23       0.95      0.95      0.95        63
          24       0.87      0.87      0.87       159
          25       0.89      0.90      0.90        52
          26       0.89      0.93      0.91        43
          27       0.97      0.97      0.97        59
          28       0.97      0.95      0.96       182
          29       0.90      0.93      0.92        60
          30       0.97      0.95      0.96        60
          31       0.95      0.92      0.93        59
          32       0.89      0.89      0.89        37
          33       0.88      0.88      0.88        16
          34       0.97      0.95      0.96       434
          35       0.97      0.91      0.94       243
          36       0.83      0.84      0.84       112

    accuracy                           0.95      4864
   macro avg       0.93      0.93      0.93      4864
weighted avg       0.95      0.95      0.95      4864


    
    '''
    

else:

    # model.load_weights('best_model.weights')
    # train_generator = data_generator(train_data+valid_data+test_data, config['batch_size'])
    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=10,
    #     callbacks=[evaluator]
    # )
    
    
    # # filepath = './datasets/Answer bot数据分类整理(0720版本)-格式修正.xlsx'
    # excel_reader=pd.ExcelFile(filepath) # 指定文件 
    # sheet_names = excel_reader.sheet_names

    # pred_data = pd.DataFrame()
    
    # for sheet in tqdm(sheet_names):
    #     if sheet != '层级结构':
    #         pred_data = pred_data.append(pd.read_excel(filepath, sheet_name=sheet).dropna(axis=0))
    
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
    # filepath = './datasets/Answer bot数据分类整理(0720版本)-格式修正.xlsx'
    # excel_reader=pd.ExcelFile(filepath) # 指定文件 
    # sheet_names = excel_reader.sheet_names

    # pred_data = pd.DataFrame()
    
    # for sheet in tqdm(sheet_names):
    #     if sheet != '层级结构':
    #         pred_data = pred_data.append(pd.read_excel(filepath, sheet_name=sheet).dropna(axis=0).drop(['label','name'],axis=1))
    
    # new_df1 = pd.read_csv('./datasets/choose_key_0722.csv',error_bad_lines=False).drop(['label','name'],axis=1)
    # new_df2 = pd.read_csv('./datasets/choose_key_0726_delta_buchong.csv',error_bad_lines=False).drop(['label','name'],axis=1)
    # new_df = pd.concat([new_df1,new_df2]).drop_duplicates(keep=False)
    
    # delta_df = pd.concat([new_df,pred_data,pred_data]).drop_duplicates(keep=False)
    
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
    
    
    
        
    