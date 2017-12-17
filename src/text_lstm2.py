# coding:utf-8
import tensorflow as tf
import sys
import os
root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.append(root_dir)
import numpy as np
import export_func
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from config import *
from tensorflow.contrib import  rnn
from optparse import OptionParser
from TextLoader import TextLoader
import time
import pypinyin
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import export_func
class text_lstm():
    def __init__(self,opt):
        self.data_path=opt.data_path
        self.model_dir=opt.save_dir
        self.rnn_size=opt.rnn_size
        self.num_epochs=opt.num_epochs
        self.learning_rate=opt.learning_rate
        self.decay_rate=opt.decay_rate
        self.num_layers=opt.num_layers
        self.seq_length=opt.seq_length
        self.keep_prob=opt.keep_prob
        self.batch_size=opt.batch_size
        self.letter_size=len(letters)
        self.le2id={}
        for i in range(len(letters)):
            self.le2id[letters[i]]=i+1
        self.data_loader=TextLoader(self.batch_size,self.seq_length)
        self.build_graph()
    def build_graph(self):
        cell=rnn.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
        self.cell=rnn.MultiRNNCell([cell]*self.num_layers)
        self.input_data1=tf.placeholder(tf.int32,[None,self.seq_length])
        self.input_data2=tf.placeholder(tf.int32,[None,self.seq_length])
        self.input_data3=tf.placeholder(tf.int32,[None,self.seq_length])
        self.input_data4=tf.placeholder(tf.int32,[None,self.seq_length])
        self.batch_size=tf.shape(self.input_data1)[0]

        with tf.variable_scope('embedding_layer'):
            weights=tf.get_variable('weights',initializer=tf.random_normal(shape=[self.letter_size+1,self.rnn_size],stddev=0.1))
            input1=tf.nn.embedding_lookup(weights,self.input_data1)
            input2=tf.nn.embedding_lookup(weights,self.input_data2)
            input3=tf.nn.embedding_lookup(weights,self.input_data3)
            input4=tf.nn.embedding_lookup(weights,self.input_data4)

            input1 = tf.split(input1, self.seq_length, 1)
            input1 = [tf.squeeze(input_, [1]) for input_ in input1]

            input2 = tf.split(input2, self.seq_length, 1)
            input2 = [tf.squeeze(input_, [1]) for input_ in input2]

            input3 = tf.split(input3, self.seq_length, 1)
            input3 = [tf.squeeze(input_, [1]) for input_ in input3]

            input4 = tf.split(input4, self.seq_length, 1)
            input4 = [tf.squeeze(input_, [1]) for input_ in input4]

        self.outputs1,last_state1=rnn.static_rnn(self.cell,input1,dtype=tf.float32,scope='rnn_layer')
        self.outputs2,last_state2=rnn.static_rnn(self.cell,input2,dtype=tf.float32,scope='rnn_layer')
        self.outputs3,last_state3=rnn.static_rnn(self.cell,input3,dtype=tf.float32,scope='rnn_layer')
        self.outputs4,last_state4=rnn.static_rnn(self.cell,input4,dtype=tf.float32,scope='rnn_layer')

        output1=self.outputs1[-1]
        output2=self.outputs2[-1]
        output3=self.outputs3[-1]
        output4=self.outputs4[-1]
        self.output1=self.outputs1[-1]
        self.cost1=tf.reduce_sum(tf.square(output1-output2),-1)
        self.cost2=tf.reduce_sum(tf.square(output3-output4),-1)

        cost=self.cost1-self.cost2
        cost=tf.clip_by_value(cost,clip_value_min=-10,clip_value_max=100)
        self.cost=tf.reduce_mean(cost)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.cost)  # Adam Optimizer
def get_loc_num():
        loc_num=TextLoader(data_path=os.path.join(data_dir,'part_loc.txt'),is_training=False).get_data()
        return loc_num
def get_asrlist(asr,le2id,seq_length):
    sub_num={}
    sub_pos={}
    lens=len(asr)
    sco=[2,3,4,5,6,7,8,9,10]
    for i in range(lens):
        for j in sco:
            if i+j<=lens:
                tmp=asr[i:i+j]
                tmp_id=[]
                tmp_pinyin=pypinyin.lazy_pinyin(tmp,0,errors='ignore')
                tmp_pinyin=' '.join(tmp_pinyin)
                for k in tmp_pinyin:
                    tmp_id.append(le2id.get(k))
                while len(tmp_id)<seq_length:
                    tmp_id.append(0)
                if len(tmp_id)>seq_length:
                    tmp_id=tmp_id[:seq_length]
                sub_pos[tmp]=(i,i+j)
                sub_num[tmp]=tmp_id

    return sub_num,sub_pos
def compute_dis(va,vb,lens):
    score=np.sqrt(np.sum(np.square(np.array(va)-np.array(vb))))/lens
    return score
def getcandi(loc_vec,sub_vec,sub_pos,cov):
    list=[]
    k=0
    while k<K:
        minv=100
        for ikey in sorted(loc_vec.keys(), key=lambda x: len(x), reverse=True):
            ivalue = loc_vec[ikey]
            for jkey in sorted(sub_vec.keys(), key=lambda x: len(x), reverse=True):
                jvalue = sub_vec[jkey]
                score=compute_dis(ivalue,jvalue,np.sqrt(len(jkey)))
                if score < minv:
                    minv=score
                    loc=ikey
                    sub=jkey
        list.append(loc)
        loc_vec.pop(loc)
        i,j=sub_pos[sub]
        for w in range(i,j):
            cov[w]+=1
        keys=[]
        for key,value in sub_vec.items():
            i,j=sub_pos[key]
            for u in range(i,j):
                if cov[u]>3:
                    if key not in keys:
                        keys.append(key)
                    cov[u]=0
        for z in keys:
            if z in sub_vec:
                sub_vec.pop(z)
        k=k+1
    return list
def train(opt):
    model=text_lstm(opt)
    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess,'/Users/guoym/Desktop/乌鲁木齐/test/text_lstm/beijing/model.ckpt')
        export_func.export(model,sess,signature_name='text_lstm',export_path=model.model_dir,version=1)
        exit()
        init=tf.global_variables_initializer()
        sess.run(init)
        minv=100
        flag=0
        for i in range(model.num_epochs):
            model.data_loader.reset_batch_pointer()
            for j in range(model.data_loader.num_batches):
                start=time.time()
                x1,x2,x3,x4=model.data_loader.next_batch()
                feed={model.input_data1:x1,model.input_data2:x2,model.input_data3:x3,model.input_data4:x4}
                cost1,cost2,train_loss,_ =sess.run([model.cost1,model.cost2,model.cost,model.optimizer], feed_dict=feed)
                print (np.mean(cost1),np.mean(cost2),train_loss)
                end = time.time()
                print ('{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.3f}'\
                .format(i * model.data_loader.num_batches + j + 1,
                        model.num_epochs * model.data_loader.num_batches,
                        i + 1,
                        train_loss,
                        end - start))
                if train_loss<=minv:
                    minv=train_loss
                    export_func.export(model,sess,signature_name='text_lstm',export_path=model.model_dir,version=(i+1)*(j+1))
                    flag=1
                    break
            if flag==1:
                break
def getvec(loc_num):
    loc_vec={}
    hostport='192.168.31.186:6000'
    host,port=hostport.split(':')
    #grpc
    channel=implementations.insecure_channel(host,int(port))
    stub=prediction_service_pb2.beta_create_PredictionService_stub(channel)
    #build request
    request= predict_pb2.PredictRequest()
    request.model_spec.name='text_lstm'
    request.model_spec.signature_name='text_lstm'
    tmp=[[0 for i in range(50)]]
    for key,order in loc_num.items():
        request.inputs['input_data1'].CopyFrom(tf.contrib.util.make_tensor_proto([order],dtype=tf.int32))
        request.inputs['input_data2'].CopyFrom(tf.contrib.util.make_tensor_proto(tmp,dtype=tf.int32))
        request.inputs['input_data3'].CopyFrom(tf.contrib.util.make_tensor_proto(tmp,dtype=tf.int32))
        request.inputs['input_data4'].CopyFrom(tf.contrib.util.make_tensor_proto(tmp,dtype=tf.int32))
        model_result=stub.Predict(request,60.0)
        output=np.array(model_result.outputs['output'].float_val)
        loc_vec[key]=output
    return loc_vec
def predict(asr,seq_length):
    le2id={}
    for i in range(len(letters)):
        le2id[letters[i]]=i+1
    loc_num=get_loc_num()
    loc_vec=getvec(loc_num)
    cov=np.zeros(len(asr))
    sub_num,sub_pos=get_asrlist(asr,le2id,seq_length)
    sub_vec=getvec(sub_num)
    list=getcandi(loc_vec,sub_vec,sub_pos,cov)
    return list

if __name__=='__main__':
    optparser=OptionParser()
    optparser.add_option("-d","--data_path",dest="data_path",type="string",default=data_path)
    optparser.add_option("-s","--save_path",dest="save_dir",type="string",default=save_dir)
    optparser.add_option("-r","--rnn_size",dest="rnn_size",type="int",default=128)
    optparser.add_option("-q","--seq_length",dest="seq_length",type="int",default=50)
    optparser.add_option("-n","--num_layers",dest="num_layers",type="int",default=1)
    optparser.add_option("-e","--num_epochs",dest="num_epochs",type="int",default=100)
    optparser.add_option("-l","--learning_rate",dest="learning_rate",type="float",default=0.01)
    optparser.add_option("-y","--decay_rate",dest="decay_rate",type="float",default=0.9)
    optparser.add_option("-c","--batch_size",dest="batch_size",type="int",default=1000)
    optparser.add_option("-k","--keep_prob",dest="keep_prob",type="float",default=1)
    opt,args=optparser.parse_args()
    #train(opt)
    asr='请给我新泰观山悦西门2月19日凌晨1:03到2:19的录像-2倍速。'
    list=predict(asr,opt.seq_length)
    for i in list:
        print (i)





