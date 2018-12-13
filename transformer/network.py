'''
Stripped down transformer network inspired architecture for MSAIC 2018
Built by Yash Bonde for team msaic_yash_sara_kuma_6223

This file contains the class for the network architecture, with built in functions for training
and operations. Any crucial step will be commented there, for more information on the model
read blog at: 
and:

Cheers!
'''

# importng the dependencies
import numpy as np
import tensorflow as tf # graph
from utils import DatasetManager, add_padding

# class

class TransformerNetwork(object):
    '''
    This is the stripped down version of Transformer network.

    In MSAIC 2018 we have to select proper paragraphs with respect to the query passed. The idea is
    attending to the important elements in query and passages and see the similarity in each one of
    them and then decide which is appropriate one. Transformer network fits here perfectly as it
    attends to both the query and passage and it's self attention picks the most important words.

    The query vector obtained in multiple stages are then also fed into the passages and also
    improves the fidelity of the outputs. We need to perform label smoothening due to disproportionate
    distribution of nagative samples.

    ========

    [GOTO: https://stackoverflow.com/a/35688187]

    The idea is that since we have an external embedding matrix, we can still use the
    functionalities available in TF to use those embedding. This will require us to store the
    embedding matrix in memory and then assign it at runtime. Function assign_embeddings() does it.
    '''
    def __init__(self,
        scope,
        save_folder,
        pad_id,
        is_training = True,
        save_freq = 10,
        dim_model = 50,
        ff_mid = 128,
        ff_mid1 = 128,
        ff_mid2 = 128,
        num_stacks = 2,
        num_heads = 5):
        '''
        Args:
            scope: scope of graph
            save_folder: folder for model saves
            pad_id: integer of <PAD>
            is_training: bool if network is in training mode
            save_freq: frequency of saving
            dim_model: same as embedding dimension
            ff_mid: dimension in middle layer of inner feed forward network
            ff_mid1: dimension in middle layer of outer feed forward network (L1)
            ff_mid2: dimension in middle layer of outer feed forward network (L2)
            num_stacks: number of stacks to use
            num_heads: number of heads in SDPA

        '''
        self.scope = scope
        self.save_folder = save_folder
        self.pad_id = pad_id
        self.is_training = is_training
        self.save_freq = save_freq
    
        self.dim_model = dim_model
        self.ff_mid = ff_mid
        self.ff_mid1 = ff_mid1
        self.ff_mid2 = ff_mid2
        self.num_stacks = num_stacks
        self.num_heads = num_heads

        self.global_step = 0


    def build_model(self, emb, seqlen, batch_size = 32, print_stack = False):
        '''
        function to build the model end to end
        '''
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.print_stack = print_stack

        with tf.variable_scope(self.scope):
            # declaring the placeholders
            self.query_input = tf.placeholder(tf.int32, [self.batch_size, self.seqlen], name = 'query_placeholder')
            self.passage_input = tf.placeholder(tf.int32, [self.batch_size, self.seqlen], name = 'passage_placeholder')
            self.target_input = tf.placeholder(tf.float32, [self.batch_size, 1], name = 'target_placeholder')
            
            # embedding matrix placeholder
            self.embedding_matrix = tf.constant(emb, name = 'embedding_matrix', dtype = tf.float32)
            
            if self.print_stack:
                print('[!] Building model...')
                print('[*] self.query_input:', self.query_input)
                print('[*] self.passage_input:', self.passage_input)
                print('[*] self.target_input:', self.target_input)
                print('[*] embedding_matrix:', self.embedding_matrix)

            # now we need to add the padding in the computation graph
            # masking
            query_mask = self.construct_padding_mask(self.query_input)   
            passage_mask = self.construct_padding_mask(self.passage_input)
            
            if self.print_stack:
                print('[*] query_mask:', query_mask)
                print('[*] passage_mask:', passage_mask)
            
            # lookup from embedding matrix
            query_emb = self.get_embedding(self.embedding_matrix, self.query_input)
            passage_emb = self.get_embedding(self.embedding_matrix, self.passage_input)
            
            if self.print_stack:
                print('[*] query_emb:', query_emb)
                print('[*] passage_emb:', passage_emb)
            
            # perform label smoothening on the labels
            # label_smooth = self.label_smoothning(self.target_input)
            label_smooth = self.target_input
            
            # model
            q_out = query_emb
            p_out = passage_emb
            for i in range(self.num_stacks):
                q_out = self.query_stack(q_in = q_out, mask = query_mask, scope = 'q_stk_{0}'.format(i))
                if self.print_stack:
                    print('[*] q_out ({0}):'.format(i), q_out)
                p_out = self.passage_stack(p_in = p_out, q_out = q_out,
                    query_mask = query_mask, passage_mask = passage_mask, scope = 'p_stk_{0}'.format(i))
                if self.print_stack:
                    print('[*] p_out ({0})'.format(i), p_out)

            # now the custom part
            ff_out = tf.layers.dense(p_out, self.ff_mid1, activation = tf.nn.relu) # (batch_size, seqlen, emb_dim)
            ff_out = tf.layers.dense(ff_out, 1, activation = tf.nn.relu) # (batch_size, seqlen, 1)
            ff_out_reshaped = tf.reshape(ff_out, [-1, seqlen]) # (batch_size, seqlen)
            self.pred = tf.layers.dense(ff_out_reshaped, 1) # (batch_size, 1)
                
            if not self.is_training:
                self.pred = tf.sigmoid(self.pred) # (batch_size, 1)
                
            if self.print_stack:
                print('[*] predictions:', self.pred)

            # loss and accuracy
            self._accuracy = tf.reduce_sum(
                tf.cast(tf.equal(self.pred, self.target_input), tf.float32)
                ) / self.batch_size

            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels = label_smooth, logits = self.pred)
                )

            optim = tf.train.AdamOptimizer(beta1 = 0.9, beta2 = 0.98, epsilon = 1e-9)
            self._train_step = optim.minimize(self._loss)
            
            if self.print_stack:
                print('[*] accuracy:', self._accuracy)
                print('[*] loss:', self._loss)
                print('[!] ... Done!')

        with tf.variable_scope(self.scope + "_summary"):
            tf.summary.scalar("loss", self._loss)
            tf.summary.scalar("accuracy", self._accuracy)
            self.merged_summary = tf.summary.merge_all()

    '''
    NETWORK FUNCTIONS
    =================

    Following functions were placed outside this file with an aim to increase the
    code value but is causing several issues, especially with the config file
    redundancy. So putting them here and increasing the model simplicity but 
    complicating the codebase.
    '''

    ##### OPERATIONAL LAYERS #####

    def get_embedding(self, emb, inp):
        '''
        get embeddings
        '''
        return tf.nn.embedding_lookup(emb, inp)

    ##### CORE LAYERS #####

    def sdpa(self, Q, K, V, mask = None):
        '''
        Scaled Dot Product Attention
        q_size = k_size = v_size
        Args:
            Q:    (num_heads * batch_size, q_size, d_model)
            K:    (num_heads * batch_size, k_size, d_model)
            V:    (num_heads * batch_size, v_size, d_model)
            mask: (num_heads * batch_size, q_size, d_model)
        '''

        qkt = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        qkt /= tf.sqrt(float(self.dim_model // self.num_heads))

        # perform masking
        qkt = tf.multiply(qkt, mask) + (1.0 - mask) * (-1e10)

        soft = tf.nn.softmax(qkt) # (num_heads * batch_size, q_size, k_size)
        soft = tf.layers.dropout(soft, training = self.is_training)
        out = tf.matmul(soft, V) # (num_heads * batch_size, q_size, d_model)

        return out

    def multihead_attention(self, query, key, value, mask = None, scope = 'attn'):
        '''
        Multihead attention with masking option
        q_size = k_size = v_size = d_model/num_heads
        Args:
            query: (batch_size, q_size, d_model)
            key:   (batch_size, k_size, d_model)
            value: (batch_size, v_size, d_model)
            mask:  (batch_size, q_size, d_model)
        '''
        with tf.variable_scope(scope):
            # linear projection blocks
            # print(query)
            Q = tf.layers.dense(query, self.dim_model, activation = tf.nn.relu)
            K = tf.layers.dense(key, self.dim_model, activation = tf.nn.relu)
            V = tf.layers.dense(value, self.dim_model, activation = tf.nn.relu)

            # split the matrix into multiple heads and then concatenate them to get
            # a larger batch size: (num_heads, q_size, d_model/nume_heads)
            Q_reshaped = tf.concat(tf.split(Q, self.num_heads, axis = 2), axis = 0)
            K_reshaped = tf.concat(tf.split(K, self.num_heads, axis = 2), axis = 0)
            V_reshaped = tf.concat(tf.split(V, self.num_heads, axis = 2), axis = 0)
            mask = tf.tile(mask, [self.num_heads, 1, 1])

            # scaled dot product attention
            sdpa_out = self.sdpa(Q_reshaped, K_reshaped, V_reshaped, mask)
            out = tf.concat(tf.split(sdpa_out, self.num_heads, axis = 0), axis = 2)

            # final linear layer
            out_linear = tf.layers.dense(out, self.dim_model)
            out_linear = tf.layers.dropout(out_linear, training = self.is_training)

        return out_linear

    def feed_forward(self, x, scope = 'ff'):
        '''
        Position-wise feed forward network, applied to each position seperately
        and identically. Can be implemented as follows
        '''
        with tf.variable_scope(scope):
            out = tf.layers.conv1d(x, filters = self.ff_mid, kernel_size = 1,
                activation = tf.nn.relu)
            out = tf.layers.conv1d(out, filters = self.dim_model, kernel_size = 1)

        return out

    def layer_norm(self, x):
        '''
        perform layer normalisation
        '''
        out = tf.contrib.layers.layer_norm(x, center = True, scale = True)
        return out

    def label_smoothning(self, x):
        '''
        perform label smoothning on the input label
        '''
        smoothed = (1.0 - self.ls_epsilon) * x + (self.ls_epsilon / vocab_size)
        return smoothed

    def construct_padding_mask(self, inp):
        '''
        Args:
            inp: Original input of word ids, shape: [batch_size, seqlen]
        Returns:
            a mask of shape [batch_size, seqlen, seqlen] where <pad> is 0 and others are 1
        '''
        seqlen = inp.shape.as_list()[1]
        mask = tf.cast(tf.not_equal(inp, self.pad_id), tf.float32)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seqlen, 1])
        return mask 

    ###### STACKS ######

    def query_stack(self, q_in, mask, scope):
        '''
        Single query stack 
        Args:
            q_in: (batch_size, seqlen, embed_size)
            mask: (batch_size, seqlen, seqlen)
        '''
        with tf.variable_scope(scope):
            multi_head = self.multihead_attention(q_in, q_in, q_in, mask)
            out = self.layer_norm(q_in + multi_head)
            out = self.layer_norm(out + self.feed_forward(out))

        return out

    def passage_stack(self, p_in, q_out, query_mask, passage_mask, scope):
        '''
        Single passage stack
        Args:
            p_in: (batch_size, seqlen, embed_size)
            q_out: output from query stack
        '''
        with tf.variable_scope(scope):
            out = self.layer_norm(p_in + self.multihead_attention(p_in, p_in, p_in, mask = passage_mask))
            out = self.layer_norm(out + self.multihead_attention(out, out, q_out, mask = query_mask, scope = 'enc_attn'))
            out = self.layer_norm(out + self.feed_forward(out))

        return out

    '''
    MODEL FUNCTIONS
    ===============

    Following functions are used for operation for the model
    '''

    def make_tf_iterators(self, q, p, l):
        '''
        since loading via tensorflow's build-in functions can significantly boost speed,
        trying to make something similar here.
        '''
        pass

    def load_model(self):
        '''
        Function to load the frozen graph of model and make the ops placeholders
        '''
        assert self.is_training == False
        pass

    def save_model(self):
        '''
        function to save the model
        '''
        save_path = self.save_folder + '/' + self.scope + '.ckpt'
        self.saver.save(self.sess, save_path)


    def print_network(self):
        '''
        Print the network in terms of 
        '''
        network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope)
        for x in network_variables:
            print(x)

    '''
    OPERATION
    =========

    Following functions are related to the operation of the model namely, training and evaluation
    '''

    def eval(self,
             queries_,
             passage_,
             query_ids,
             file_path):
        '''
        Function to run the model and store the results in file_path
        Args:
            queries_: numpy array for queries
            passage_: numpy array for passages
            query_ids: IDs for corresponding queries, needed to store the results
            file_path: location of file to store results
        '''
        
        pass

    def train(self,
              queries_,
              passage_,
              label_,
              num_epochs = 50,
              val_split = 0.1):
        '''
        This is the function used to train the model.
        Args:
            queries_: numpy array for queries
            passage_: numpy array for passages
            label_: numpy array for labels
        '''
        if not self.is_training:
            raise Exception("Config not setup for training,", self.is_training)

        # make the dataset manager to handle the datasets
        dm = DatasetManager(queries_, passage_, label_)

        # establish the saver 
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # train logs
        train_loss = []
        train_acc = []

        for ep in range(num_epochs):
            batch_loss = []
            batch_accuracy = []

            # iterate over all the batches
            for batch_num in range(dm.get_num_batches(self.batch_size)):
                # for each epoch, go over the entire dataset once
                b_query, b_passage, b_label = dm.get_batch(queries_, passage_, label_, self.batch_size)

                # pad the sequences
                b_query = add_padding(b_query, self.pad_id, self.seqlen)
                b_passage = add_padding(b_passage, self.pad_id, self.seqlen)

                # reshape
                b_label = np.reshape(b_label, [-1, 1])

                # print stats if
                if ep == 0 and batch_num == 0:
                    print('[*] Batch Shapes:')
                    print('b_query:', b_query.shape)
                    print('b_passage:', b_passage.shape)
                    print('b_label:', b_label.shape)

                # operate
                b_ops = [self._loss, self._accuracy, self._train_step]
                feed_dict = {self.query_input: b_query, self.passage_input: b_passage, self.target_input: b_label}
                b_loss, b_acc, _ = self.sess.run(b_ops, feed_dict)

                batch_loss.append(b_loss)
                batch_accuracy.append(b_acc)

                if self.global_step != 0 and self.global_step % self.save_freq == 0:
                    self.save_model()

                self.global_step += 1

            # once all the batches are done
            mean_loss = np.mean(batch_loss)
            mean_acc = np.mean(batch_accuracy)

            train_loss.append(mean_loss)
            train_acc.append(mean_acc)

            '''
            == Add to Tensorboard output ==
            Add the stats to tensorboard output. Note that there alead is a function self.merge_all but I don't know
            how to use this.
            '''

        print(train_loss)
        print(train_acc)
