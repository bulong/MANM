#-*-coding:utf-8 -*-


from keras.layers import Input,LSTM,Dense,GRU,Bidirectional,Multiply,Permute
from keras.models import Model,load_model
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf

import keras
from keras.layers import Layer


class Attention(Layer):

    def __init__(self,**kwargs):

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape:[(batch_size,max_step,2 * encoder_size),(batch_size,max_step,2 * encoder_size)]
        assert isinstance(input_shape, list)

        # Concat Attention
        self.Wc1 = self.add_weight(name='Wc1',
                                   shape=(input_shape[0][2], input_shape[0][1]),
                                   initializer='uniform',
                                   trainable=True)
        self.Wc2 = self.add_weight(name='Wc2',
                                   shape=(input_shape[0][2], input_shape[0][1]),
                                   initializer='uniform',
                                   trainable=True)
        self.vc = self.add_weight(name='vc',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)
        # Bilinear Attention
        self.Wb = self.add_weight(name='Wb',
                                   shape=(input_shape[0][2], input_shape[0][2]),
                                   initializer='uniform',
                                   trainable=True)
        # Dot Attention,Wd共享参数
        self.Wd = self.add_weight(name='Wd',
                                  shape=(input_shape[0][2], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.vd = self.add_weight(name='vd',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)

        # Minus Attention,Wm共享参数
        self.Wm = self.add_weight(name='Wm',
                                  shape=(input_shape[0][2], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.vm = self.add_weight(name='vm',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)



        super(Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        q_sentence_output, p_sentence_output = x

        # Concat Attention
        whq = K.dot(q_sentence_output, self.Wc1)  # 每个banch内矩阵乘法，shape:(batch_size,max_step,max_step)
        whp = K.dot(p_sentence_output, self.Wc2)  # shape:(batch_size,max_step,max_step)
        sc = tf.multiply(tf.tanh(tf.add(whp, tf.transpose(whq, perm=[0, 2, 1]))), self.vc)  # (batch_size,max_step,max_step)
        ac = tf.nn.softmax(sc)  # 按列softmax，(batch_size,max_step,max_step)
        qc = K.batch_dot(ac, q_sentence_output)  # shape:(batch_size,max_step,2 * encoder_size)

        # Bilinear Attention
        sb = K.batch_dot(K.dot(p_sentence_output, self.Wb),
                         tf.transpose(q_sentence_output, [0, 2, 1]))  # (batch_size,max_step,max_step)
        ab = tf.nn.softmax(sb)  # 按列softmax，(max_step,max_step)
        qb = K.batch_dot(ab, q_sentence_output)  # shape:(batch_size,max_step,2 * encoder_size)

        # Dot Attention,Wd共享参数
        q_p_dot = tf.expand_dims(q_sentence_output, axis=1) * tf.expand_dims(p_sentence_output, axis=2)

        sd = tf.multiply(tf.tanh(K.dot(q_p_dot, self.Wd)), self.vd)  # (batch_size,max_step,max_step)
        sd = tf.squeeze(sd, axis=-1)
        ad = tf.nn.softmax(sd)  # 按列softmax，(batch_size,max_step,max_step)
        qd = K.batch_dot(ad, q_sentence_output)  # shape:(batch_size,max_step,2 * encoder_size)

        # Minus Attention,Wm共享参数
        q_p_minus = tf.expand_dims(q_sentence_output, axis=1) - tf.expand_dims(p_sentence_output, axis=2)

        sm = tf.multiply(tf.tanh(K.dot(q_p_minus, self.Wm)), self.vm)
        sm = tf.squeeze(sm, axis=-1)
        am = tf.nn.softmax(sm)  # 按列softmax，(batch_size,max_step,max_step)
        qm = K.batch_dot(am, q_sentence_output)  # shape:(batch_size,max_step,2 * encoder_size)


        return [qc,qb,qd,qm]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [input_shape[0]]*4

class Inside_Aggregation(Layer):

    def __init__(self,**kwargs):

        super(Inside_Aggregation, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape:[(batch_size,max_step,2 * encoder_size),(batch_size,max_step,2 * encoder_size)]
        assert isinstance(input_shape, list)
        # Concat Attention
        self.Wg = self.add_weight(name='Wg',
                                   shape=(2*input_shape[0][2], 1),
                                   initializer='uniform',
                                   trainable=True)

        super(Inside_Aggregation, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        qj, p_sentence_output = x

        xj = tf.concat([qj, p_sentence_output], axis=2)

        gj = tf.sigmoid(K.dot(xj, self.Wg))
        xj_ = gj * xj

        return xj_

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return (input_shape[0][0],input_shape[0][1],2*input_shape[0][2])

class Mix_Aggregation(Layer):

    def __init__(self,shape,**kwargs):
        self.shape = shape
        super(Mix_Aggregation, self).__init__(**kwargs)


    def build(self, input_shape):
        #input_shape:[(batch_size,max_step,2 * encoder_size),(batch_size,max_step,2 * encoder_size)]
        assert isinstance(input_shape, list)
        # Concat Attention
        self.W1 = self.add_weight(name='W1',
                                   shape=(input_shape[0][2], 1),
                                   initializer='uniform',
                                   trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)

        self.va = self.add_weight(name='va',
                                  shape=(1, 4),
                                  initializer='uniform',
                                  trainable=True)
        self.v = self.add_weight(name='v',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)

        super(Mix_Aggregation, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        hc, hb, hd, hm = x

        h_cat_tmp_= keras.layers.concatenate([hc, hb, hd, hm], axis=2)

        h_cat = keras.layers.Reshape((self.shape[1], 4, self.shape[2]))(h_cat_tmp_)  # shape:(batch_size,max_step,4,2 * encoder_size),每一step有4个h


        w1_h = K.dot(h_cat, self.W1)  # 每个banch内矩阵乘法，shape:(batch_size,max_step,4,1)
        # w1_h_ = K.reshape(w1_h, (batch_size, max_step, 4))
        w1_h_ = tf.squeeze(w1_h, axis=-1)

        # w2_va = K.dot(W2, va)  # shape:(batch_size,max_step,max_step)
        s = tf.multiply(tf.add(w1_h_, K.dot(self.W2, self.va)), self.v)
        a = tf.nn.softmax(s)
        x = K.batch_dot(a, h_cat)

        return x

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return (input_shape[0][0],input_shape[0][1],input_shape[0][2])

class Prediction(Layer):

    def __init__(self,**kwargs):

        super(Prediction, self).__init__(**kwargs)


    def build(self, input_shape):
        #input_shape:[(batch_size,max_step,2 * encoder_size),(batch_size,max_step,2 * encoder_size)]
        assert isinstance(input_shape, list)
        # Concat Attention
        self.Wq1 = self.add_weight(name='Wq1',
                                   shape=(input_shape[0][2], 1),
                                   initializer='uniform',
                                   trainable=True)
        self.Wq2 = self.add_weight(name='Wq2',
                                  shape=(input_shape[0][1], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.Wp1 = self.add_weight(name='Wp1',
                                   shape=(input_shape[0][2], 1),
                                   initializer='uniform',
                                   trainable=True)
        self.Wp2 = self.add_weight(name='Wp2',
                                   shape=(input_shape[0][2],input_shape[0][1]),
                                   initializer='uniform',
                                   trainable=True)

        self.v = self.add_weight(name='v',
                                 shape=(input_shape[0][1], 1),
                                 initializer='uniform',
                                 trainable=True)

        self.vq = self.add_weight(name='vq',
                                  shape=(1, 1),
                                  initializer='uniform',
                                  trainable=True)


        super(Prediction, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        h,q_sentence_output = x

        sq = tf.multiply(tf.add(K.dot(q_sentence_output, self.Wq1), K.dot(self.Wq2, self.vq)), self.v)
        sq = tf.squeeze(sq, axis=-1)
        aq = tf.nn.softmax(sq)  # 按列softmax，(batch_size,max_step)
        rq = K.batch_dot(aq,q_sentence_output)  # 求得q_sentence_output所有步长的加权平均(句子q的向量表示),shape:(batch_size,2 * encoder_size)


        m1_ = K.dot(h, self.Wp1)  # 每个banch内矩阵乘法，shape:(batch_size,max_step,1)

        m2_ = tf.expand_dims(K.dot(rq, self.Wp2), axis=-1)  # shape:(batch_size,max_step,1)
        sr = tf.multiply(tf.tanh(tf.add(m1_, m2_)), self.v)  # (batch_size,max_step,max_step)
        sr = tf.squeeze(sr, axis=-1)
        ar = tf.nn.softmax(sr)  # 按列softmax，(batch_size,max_step,max_step)
        rp = K.batch_dot(ar, h)  # shape:(batch_size,2 * encoder_size),句子p的向量表示

        return rp

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return (input_shape[0][0],input_shape[0][2])


def create_model(embedding_size, encoder_size,max_step,batch_size):

    # 训练阶段
    # encoder
    q_sentence_input = Input(shape=(max_step,embedding_size),dtype='float32',name='q_sentence_input')
    p_sentence_input = Input(shape=(max_step,embedding_size),dtype='float32',name='p_sentence_input')

    # encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    q_sentence_encoder = Bidirectional(GRU(encoder_size, return_sequences=True),merge_mode='concat')
    p_sentence_encoder = Bidirectional(GRU(encoder_size, return_sequences=True), merge_mode='concat')
    # n_units为GRU单元中每个门的神经元的个数，return_sequences设为True时返回所有时刻输出
    #shape:(batch_size,max_step,2 * encoder_size)
    q_sentence_output = q_sentence_encoder(q_sentence_input)
    p_sentence_output = p_sentence_encoder(p_sentence_input)

    qc, qb, qd, qm = Attention()([q_sentence_output,p_sentence_output])
    # Inside Aggregation
    xc_ = Inside_Aggregation()([qc, p_sentence_output])
    xb_ = Inside_Aggregation()([qb, p_sentence_output])
    xd_ = Inside_Aggregation()([qd, p_sentence_output])
    xm_ = Inside_Aggregation()([qm, p_sentence_output])

    in_agg_encoder = Bidirectional(GRU(encoder_size, return_sequences=True), merge_mode='concat')

    #shape:(batch_size,max_step,2*encoder_size)
    hc = in_agg_encoder(xc_)
    hb = in_agg_encoder(xb_)
    hd = in_agg_encoder(xd_)
    hm = in_agg_encoder(xm_)

    # mixed aggregation,shape:(batch_size,max_step,2*encoder_size)
    x = Mix_Aggregation(K.int_shape(hc))([hc, hb, hd, hm])
    mix_agg_encoder = Bidirectional(GRU(encoder_size, return_sequences=True), merge_mode='concat')
    h = mix_agg_encoder(x)

    # Prediction Layer
    rp = Prediction()([h,q_sentence_output])
    prediction = Dense(1, activation='sigmoid', input_shape=(None, 2 * encoder_size))
    pred = prediction(rp)

    # 生成的训练模型,第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出
    model = Model(inputs = [q_sentence_input, p_sentence_input], outputs = pred)

    return model



