# coding:utf-8
import tensorflow as tf
import sys
import os
import numpy as np
import pickle
import export_func
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib import  rnn
from optparse import OptionParser
from TextLoader import TextLoader
import time
import pypinyin
import numpy as np
from utils import *
import export_func

root_dir='/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.append(root_dir)
letters = [' ','a','b','c','d','e','f','j','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

config_name='config.ini'
config=Config()(config_name)['model_parameter']
config_p=Config()(config_name)['predict_parameter']

class text_lstm():
    def __init__(self,opt):
        self.model_dir = opt['save_dir']
        self.rnn_size = opt['rnn_size']
        self.num_epochs = opt['num_epochs']
        self.num_layers = opt['num_layers']
        self.batch_size = opt['batch_size']
        self.letter_size = len(opt['letters'])
        self.build_graph()
    def build_graph(self):

        cell = tf.contrib.rnn.LSTMCell(self.rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        self.cell = rnn.MultiRNNCell([cell])

        self.input_data1 = tf.placeholder(tf.int32,[None,None])
        self.input_data2 = tf.placeholder(tf.int32,[None,None])
        self.input_data3 = tf.placeholder(tf.int32,[None,None])
        self.input_data4 = tf.placeholder(tf.int32,[None,None])
        self.input_length1 = tf.placeholder(tf.int32,[None])
        self.input_length2 = tf.placeholder(tf.int32,[None])
        self.input_length3 = tf.placeholder(tf.int32,[None])
        self.input_length4 = tf.placeholder(tf.int32,[None])


        with tf.variable_scope('embedding_layer'):
            weights = tf.get_variable('weights',initializer=tf.random_normal(shape=[self.letter_size,self.rnn_size],stddev=0.1))
        input1 = tf.nn.embedding_lookup(weights,self.input_data1)
        input2 = tf.nn.embedding_lookup(weights,self.input_data2)
        input3 = tf.nn.embedding_lookup(weights,self.input_data3)
        input4 = tf.nn.embedding_lookup(weights,self.input_data4)


        _,output1 = tf.nn.dynamic_rnn(self.cell,input1,sequence_length=self.input_length1,dtype=tf.float32)
        _,output2 = tf.nn.dynamic_rnn(self.cell,input2,sequence_length=self.input_length2,dtype=tf.float32)
        _,output3 = tf.nn.dynamic_rnn(self.cell,input3,sequence_length=self.input_length3,dtype=tf.float32)
        _,output4 = tf.nn.dynamic_rnn(self.cell,input4,sequence_length=self.input_length4,dtype=tf.float32)

        self.output1 = output1[-1].c
        self.output2 = output2[-1].c
        self.output3 = output3[-1].c
        self.output4 = output4[-1].c

        self.cost1 = tf.reduce_sum(tf.square(self.output1-self.output2),-1)
        self.cost2 = tf.reduce_sum(tf.square(self.output3-self.output4),-1)

        cost = self.cost1-self.cost2
        cost = tf.clip_by_value(cost,clip_value_min=-10,clip_value_max=100)
        self.cost = tf.reduce_mean(cost)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.cost)  # Adam Optimizer

def train(opt):
    model = text_lstm(opt)
    loader = TextLoader(opt['batch_size'],opt['seq_length'],opt['data_path'])
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # saver.restore(sess,opt.save_dir)
        # export_func.export(model,sess,signature_name='text_lstm',export_path=model.model_dir,version=1)
        # exit()
        init = tf.global_variables_initializer()
        sess.run(init)
        minv = 100
        for i in range(model.num_epochs):
            loader.reset_batch_pointer()
            for j in range(loader.num_batches):
                start = time.time()
                x1,x2,x3,x4,l1,l2,l3,l4 = loader.next_batch()
                feed = {model.input_data1:x1,
                      model.input_data2:x2,
                      model.input_data3:x3,
                      model.input_data4:x4,
                      model.input_length1:l1,
                      model.input_length2:l2,
                      model.input_length3:l3,
                      model.input_length4:l4}
                cost1,cost2,train_loss,_  = sess.run([model.cost1,model.cost2,model.cost,model.optimizer], feed_dict=feed)
                print (np.mean(cost1),np.mean(cost2),train_loss)
                end = time.time()
                print ('{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.3f}'\
                .format(i * loader.num_batches + j + 1,
                        model.num_epochs * loader.num_batches,
                        i + 1,
                        train_loss,
                        end - start))
                if train_loss <= minv:
                    minv = train_loss
                    saver.save(sess,save_path=opt.save_dir)

def predict(asr,seq_length):
    le2id={}
    for i in range(len(letters)):
        le2id[letters[i]]=i
    if not os.path.exists(os.path.join(root_dir,config_p['candi_loc'])):
        loc_num,lens = get_loc_num(os.path.join(root_dir,config['loc_path']))
        loc_vec = getvec(config_p,loc_num,lens)
        with open(os.path.join(root_dir,config_p['candi_loc']),'wb') as f:
            pickle.dump(loc_vec,f)
    else:
        with open(os.path.join(root_dir,config_p['candi_loc']),'rb') as f:
            loc_vec=pickle.load(f)

    cov = np.zeros(len(asr))
    sub_num,sub_pos,sub_lens = get_asrlist(asr,le2id,seq_length)
    sub_vec = getvec(config_p,sub_num,sub_lens)
    list = getcandi(loc_vec,sub_vec,sub_pos,cov)
    return list

if __name__=='__main__':
    # train(config)
    asr='安宁渠镇东村三队丁字路口'
    list=predict(asr,int(config['seq_length']))
    for i in list:
        print (i)





