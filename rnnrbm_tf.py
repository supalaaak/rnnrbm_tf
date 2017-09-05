# This is a modified version of Nicolas Boulanger-Lewandowski's RNN-RBM using Tensorflow (machinne learning software libary) instead of Theano
# More detaila of Nicolas's work  http://deeplearning.net/tutorial/rnnrbm.html

from __future__ import print_function

import glob
import os
import sys
import tensorflow as tf

import numpy

configfile = '~/Downloads/midi'
sys.path.append(os.path.dirname(os.path.expanduser(configfile)))


from utils import midiread, midiwrite



def build_rbm(v, W, bv, bh, k):
    
    def gibbs_step(v):
        mean_h=tf.sigmoid(tf.add(tf.matmul(v,W),bh))
        h=tf.where(numpy.random.uniform(size=(mean_h.shape[0].value,(mean_h.shape[1].value))) - mean_h < 0,tf.ones(mean_h.get_shape()), tf.zeros(mean_h.get_shape()))
        mean_v=tf.sigmoid(tf.add(tf.matmul(h,tf.transpose(W)),bv))
        v=tf.where(numpy.random.uniform(size=(mean_v.shape[0].value,(mean_v.shape[1].value))) - mean_v < 0,tf.ones(mean_v.get_shape()), tf.zeros(mean_v.get_shape()))
        return mean_v, v
    
    
    chain=tf.scan(lambda x, _: gibbs_step(x)[1], elems=tf.range(k), initializer=v)
    
    v_sample = chain[-1]
    v_temp= gibbs_step(v_sample)
    mean_v = v_temp[0]
    
    monitor = tf.reduce_mean(tf.reduce_sum(-v*tf.log(mean_v)-(1-v)*tf.log(1-mean_v), reduction_indices=[1]))
    def free_energy(v):
	     return  -tf.reduce_mean(tf.matmul(v,tf.transpose(bv)) + tf.reduce_sum(tf.exp(1+tf.add(tf.matmul(v,W),bh)), reduction_indices=[1], keep_dims=True))
    cost = (free_energy(v) - free_energy(v_sample))
    return v_sample, cost, monitor

def initialize_parameters(n_visible, n_hidden, n_hidden_recurrent):
    sbv= tf.get_variable("bv", [1,n_visible], initializer = tf.zeros_initializer())
    sbh= tf.get_variable("bh", [1,n_hidden], initializer = tf.zeros_initializer())
    sbu= tf.get_variable("bu", [1,n_hidden_recurrent], initializer = tf.zeros_initializer())
    sW = tf.get_variable("W", [n_visible,n_hidden], initializer = tf.contrib.layers.xavier_initializer())
    sWuh = tf.get_variable("Wuh", [n_hidden_recurrent,n_hidden], initializer = tf.contrib.layers.xavier_initializer())
    sWuv = tf.get_variable("Wuv", [n_hidden_recurrent,n_visible], initializer = tf.contrib.layers.xavier_initializer())
    sWvu = tf.get_variable("Wvu",[n_visible,n_hidden_recurrent], initializer = tf.contrib.layers.xavier_initializer())
    sWuu = tf.get_variable("Wuu", [n_hidden_recurrent,n_hidden_recurrent], initializer = tf.contrib.layers.xavier_initializer())
    params = { "W": sW,"bv": sbv,"bh": sbh,"Wuh": sWuh,"Wuv": sWuv,"Wvu": sWvu,"Wuu": sWuu,"bu": sbu}
    return params


def build_rnnrbm(v, parameters):
    W = parameters['W']
    bv = parameters['bv']
    bh = parameters['bh']
    Wuh = parameters['Wuh']
    Wuv = parameters['Wuv']
    Wvu = parameters['Wvu']
    Wuu = parameters['Wuu']
    bu = parameters['bu']
    n_visible=W.shape[0].value
    n_hidden=W.shape[1].value
    n_hidden_recurrent=Wuu.shape[0].value
    u0 = tf.zeros((1,n_hidden_recurrent))
    initial_hidden = [u0, tf.zeros((1,n_visible)), tf.zeros((1,n_hidden))]
    
    def forward_recurrence(prev, v_t):
        u_tm1=prev[0]
        bv_st = bv+tf.matmul(u_tm1,Wuv)
        bh_st = bh+tf.matmul(u_tm1,Wuh)
        u_t = tf.tanh(bu + tf.matmul(tf.reshape(v_t,[1, n_visible]), Wvu) + tf.matmul(u_tm1, Wuu))
        return [u_t, bv_st, bh_st]
    
    if v.shape[0].value==None:
        v_sample, cost, monitor= build_rbm(tf.zeros((1,W.shape[0])), W, bv, bh, k=1)
    
    else:
        (u_t, bv_t, bh_t)=tf.scan(fn=forward_recurrence,elems=[v], initializer=initial_hidden)
        #uu_t=tf.reshape(u_t,[u_t.shape[0].value,u_t.shape[2].value])
        bbv_t=tf.reshape(bv_t,[bv_t.shape[0].value,bv_t.shape[2].value])
        bbh_t=tf.reshape(bh_t,[bh_t.shape[0].value,bh_t.shape[2].value])
        v_sample, cost, monitor= build_rbm(v, W, bbv_t[:], bbh_t[:], k=15)
    return (v_sample, cost, monitor)


def gen_rnnrbm(parameters, nsteps=200):

    W = parameters['W']
    bv = parameters['bv']
    bh = parameters['bh']
    Wuh = parameters['Wuh']
    Wuv = parameters['Wuv']
    Wvu = parameters['Wvu']
    Wuu = parameters['Wuu']
    bu = parameters['bu']
    
    n_visible=W.shape[0].value
    
    def backward_recurrence(prev, v_t):
        
        u_tm1=prev[0]
        bvs_t = bv+tf.matmul(u_tm1,Wuv)
        bhs_t = bh+tf.matmul(u_tm1,Wuh)
        vv_t, _, _ = build_rbm(tf.zeros((1,n_visible)), W, bvs_t, bhs_t, k=25)
        
        uu_t = tf.tanh(bu + tf.matmul(vv_t, Wvu) + tf.matmul(u_tm1, Wuu))
        
        return [uu_t,vv_t]
    
    initial_gen=[tf.zeros((1,n_hidden_recurrent)), tf.zeros((1,n_visible))]
    input_gen=tf.zeros((nsteps,1,n_visible))
    (u_t, v_t)=tf.scan(fn=backward_recurrence,elems=[input_gen],initializer=initial_gen)

    return v_t


class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences.'''

    def __init__(
        self,
        n_hidden=150,
        n_hidden_recurrent=100,
        lr=0.001,
        n_visible=88,
        r=(21, 109),
        dt=0.3
    ):
        '''Constructs and compiles Theano functions for training and sequence
        generation.

        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        r : (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        dt : float
            Sampling period when converting the MIDI files into piano-rolls, or
            equivalently the time difference between consecutive time steps.'''

        self.r = r
        self.dt = dt



    def train(self, files, batch_size=100, num_epochs=1):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        files : list of strings
            List of MIDI files that will be loaded as piano-rolls for training.
        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time.'''

        assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
    
        dataset = [midiread(f, r, dt).piano_roll.astype(numpy.float32) for f in files]
        
        v=tf.placeholder(tf.float32, shape=(None,n_visible))
        
        with tf.variable_scope("model") as scope:
            try:
                para=initialize_parameters(n_visible, n_hidden, n_hidden_recurrent)
            except ValueError:
                scope.reuse_variables()
                para=initialize_parameters(n_visible, n_hidden, n_hidden_recurrent)
        
        (v_sample, cost, monitor) = build_rnnrbm(v, para)
    

        optimizer =tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(monitor)
        init = tf.global_variables_initializer()

        try:
            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(num_epochs):
                    numpy.random.shuffle(dataset)
                    costs = []
                    for s, sequence in enumerate(dataset):
                        for i in range(0, len(sequence), batch_size):
                            _ , cost2=sess.run([optimizer, monitor], feed_dict={v:sequence[i:i + batch_size]})
                            costs.append(cost2)
                    print('Epoch %i/%i' % (epoch + 1, num_epochs))
                    print(numpy.mean(costs))
                    sys.stdout.flush()
                W1=gen_rnnrbm(para, nsteps=200)
                piano_roll = sess.run(W1)
                #piano_roll=tf.reshape(piano_roll,[piano_roll.shape[0].value,piano_roll.shape[2].value])
                print(piano_roll)
                midiwrite('sample1.mid', piano_roll, self.r, self.dt)
        except KeyboardInterrupt:
            print('Interrupted by user.')

            



def test_rnnrbm(batch_size=100, num_epochs=200):
    model = RnnRbm()
    re = os.path.join(os.path.split(os.path.dirname('__file__'))[0],'data', 'Nottingham', 'train', '*.mid')
    model.train(glob.glob(re),batch_size=batch_size, num_epochs=num_epochs)
    #model.generate('sample1.mid',parameters, show=False)
    return model

if __name__ == '__main__':
    model = test_rnnrbm()
    #model.generate('sample1.mid')
    pylab.show()
