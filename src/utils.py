from configparser import ConfigParser
from TextLoader import *
import tensorflow as tf
import numpy as np
import sys
import os
import pypinyin
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from sklearn.metrics.pairwise import  pairwise_distances

class Config(ConfigParser):
    def __init__(self,*args,**kwargs):
        super(Config,self).__init__(*args,**kwargs)
    def __call__(self, filename,encoding=None):
        self.read(filename,encoding=encoding)
        return self

def get_loc_num(filename):
    loc_num,lens = TextLoader(data_path=filename,is_training=False).get_data()
    return loc_num,lens

def getvec(config,loc_num,lens):
    loc_vec = {}

    channel = implementations.insecure_channel(config['TF_SERVING_HOST'],int(config['TF_SERVING_PORT']))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    #build request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'text_lstm'
    request.model_spec.signature_name = 'text_lstm'
    tmp=[[0 for i in range(110)]]
    count=0

    for key,order in loc_num.items():
        request.inputs['input_data1'].CopyFrom(tf.contrib.util.make_tensor_proto([order],dtype=tf.int32))
        request.inputs['input_data2'].CopyFrom(tf.contrib.util.make_tensor_proto(tmp,dtype=tf.int32))
        request.inputs['input_data3'].CopyFrom(tf.contrib.util.make_tensor_proto(tmp,dtype=tf.int32))
        request.inputs['input_data4'].CopyFrom(tf.contrib.util.make_tensor_proto(tmp,dtype=tf.int32))
        request.inputs['input_length1'].CopyFrom(tf.contrib.util.make_tensor_proto([lens[count]],dtype=tf.int32))
        request.inputs['input_length2'].CopyFrom(tf.contrib.util.make_tensor_proto(0,dtype=tf.int32))
        request.inputs['input_length3'].CopyFrom(tf.contrib.util.make_tensor_proto(0,dtype=tf.int32))
        request.inputs['input_length4'].CopyFrom(tf.contrib.util.make_tensor_proto(0,dtype=tf.int32))
        model_result=stub.Predict(request,200.0)
        output=np.array(model_result.outputs['output'].float_val)
        loc_vec[key]=output
        count+=1
    return loc_vec

def get_asrlist(asr,le2id,seq_length):
    sub_num={}
    sub_pos={}
    sub_lens=[]
    lens=len(asr)
    sco=[lens]
    for i in range(lens):
        for j in sco:
            if i+j<=lens:
                tmp=asr[i:i+j]
                tmp_id=[]
                tmp_pinyin=pypinyin.lazy_pinyin(tmp,0,errors='ignore')
                tmp_pinyin=' '.join(tmp_pinyin)
                for k in tmp_pinyin:
                    tmp_id.append(le2id.get(k))
                sub_lens.append(len(tmp_id))
                while len(tmp_id)<seq_length:
                    tmp_id.append(0)
                if len(tmp_id)>seq_length:
                    tmp_id=tmp_id[:seq_length]
                sub_pos[tmp]=(i,i+j)
                sub_num[tmp]=tmp_id

    return sub_num,sub_pos,sub_lens

def getcandi(loc_vec,sub_vec,sub_pos,cov):
    list = []
    loc_vec_arr = [loc_vec[ikey] for ikey in sorted(loc_vec.keys(),key=lambda x:len(x),reverse=True)]
    loc_vec_np = np.array(loc_vec_arr)
    sub_vec_arr = [sub_vec[ikey] for ikey in sorted(sub_vec.keys(),key=lambda x:len(x),reverse=True)]
    sub_vec_np = np.array(sub_vec_arr)
    mn_matrix = pairwise_distances(loc_vec_np,sub_vec_np,metric='euclidean',n_jobs=-1)

    s_loc_vec = sorted(loc_vec.keys(),key=lambda x:len(x),reverse=True)
    s_sub_vec = sorted(sub_vec.keys(),key=lambda x:len(x),reverse=True)
    K=4
    k=0
    while k<K:

        imin,jmin = np.unravel_index(mn_matrix.argmin(),mn_matrix.shape)#返回下标的tuple
        loc = s_loc_vec[imin]
        sub = s_sub_vec[jmin]

        list.append(loc)

        ki = s_loc_vec.index(loc)
        for i in range(mn_matrix.shape[1]):
            mn_matrix[ki][i] = float("inf")

        loc_vec.pop(loc)

        i,j = sub_pos[sub]
        for w in range(i,j):
            cov[w] += 1

        keys = []
        for key, value in sub_vec.items():
            i, j = sub_pos[key]
            for u in range(i,j):
                if cov[u] > 3:
                    if key not in keys:
                        keys.append(key)
                    cov[u]=0

        for z in keys:
            if z in sub_vec:
                sub_vec.pop(z)
                ki = s_sub_vec.index(z)
                for i in range(mn_matrix.shape[0]):
                    mn_matrix[i][ki] = float("inf")
        k += 1
    return list

