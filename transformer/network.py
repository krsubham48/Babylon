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
        model_name,
        save_folder,
        save_freq = 10,
        is_training = True,
        dim_model = 50,
        ff_mid = 128,
        num_stacks = 2,
        num_heads = 6):
        '''
        Args:
            scope: scope of graph
            model_name: name for model
            save_folder: folder for model saves
            save_freq: frequency of saving
            is_training: bool if network is in training mode
            dim_model: same as embedding dimension
            ff_mid: dimension in middle layer of feed forward network
            num_stacks: number of stacks to use
            num_heads: number of heads in SDPA

        '''
        self.scope = scope
        self.model_name = model_name
        self.is_training = is_training
        self.save_folder = save_folder
        self.save_freq = save_freq

        self.global_step = 0

    def build_model(self, emb, seqlen, batch_size = 32):
        '''
        function to build the model end to end
        '''
        self.batch_size = batch_size
        self.seqlen = seqlen

        with tf.variable_scope(self.scope):
            # declaring the placeholders
            self.query_input = tf.placeholder(tf.int32, [self.batch_size, self.seqlen], name = 'query_placeholder')
            self.passage_input = tf.placeholder(tf.int32, [self.batch_size, self.seqlen], name = 'passage_placeholder')
            self.target_input = tf.placeholder(tf.float32, [self.batch_size, 1], name = 'target_placeholder')

            # embedding matrix placeholder
            self.embedding_matrix = tf.constant(emb, name = 'embedding_matrix')

            # now we need to add the padding in the computation graph
            query_emb = self.get_embedding(self.embedding_matrix, self.query_input)
            passage_emb = self.get_embedding(self.embedding_matrix, self.passage_input)

            # perform label smoothening on the labels
            label_smooth = self.label_smoothening(self.target_input)

            q_out = self.query_input
            p_out = self.passage_input
            for i in range(self.num_stacks):
                q_out = self.query_stack(q_in = q_out, mask = input_mask, scope = 'q_stk_{i}')
                p_out = self.passage_stack(p_in = p_out, q_out = q_out,
                    input_mask = input_mask, target_mask = target_mask, scope = 'p_stk_{i}')

            # now the custom part
            ff_out = tf.layers.dense(p_out, self.FINAL_ff_mid1, activation = tf.nn.relu)
            ff_out = tf.layers.dense(ff_out, self.FINAL_ff_mid2)
            logits = tf.layers.dense(ff_out, 1) # (batch_size, 1)
            if not self.is_training:
                self.pred = tf.sigmoid(logits) # (batch_size, 1)

            # loss and accuracy
            self._accuracy = tf.reduce_sum(
                tf.cast(tf.equal(self.pred, self.target_input), tf.float32)
                ) / self.batch_size

            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels = label_smooth, logits = logits)
                )

            optim = tf.train.AdamOptimizer(beta1 = 0.9, beta2 = 0.98, epsilon = 1e-9)
            self._train = optim.minimize(self._loss)

        with tf.variable_scope(self.model_name + "_summary"):
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

    def get_embedding(emb, inp):
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
        qkt /= tf.sqrt(np.float32(self.dim_model // self.num_heads))

        if mask:
            # perform masking
            qkt = tf.multiply(qkt, mask) + (1.0 - mask) * (-1e10)

        soft = tf.nn.softmax(qkt) # (num_heads * batch_size, q_size, k_size)
        soft = tf.layers.dropout(soft, training = self.is_training)
        out = tf.matmul(soft, V) # (num_heads * batch_size, q_size, d_model)

        return out

    def multihead_attention(self, query, key, value, mask = None, scope = 'attention'):
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
            Q = tf.layers.dense(query, self.dim_model, activation = tf.nn.relu)
            K = tf.layers.dense(key, self.dim_model, activation = tf.nn.relu)
            V = tf.layers.dense(value, self.dim_model, activation = tf.nn.relu)

            # split the matrix into multiple heads and then concatenate them to get
            # a larger batch size: (num_heads, q_size, d_model/nume_heads)
            Q_reshaped = tf.concat(tf.split(Q, self.num_heads, axis = 2), axis = 0)
            K_reshaped = tf.concat(tf.split(K, self.num_heads, axis = 2), axis = 0)
            V_reshaped = tf.concat(tf.split(V, self.num_heads, axis = 2), axis = 0)
            if mask:
                mask = tf.tile(mask, [self.num_heads, 1, 1])

            # scaled dot product attention
            sdpa_out = sdpa(Q_reshaped, K_reshaped, V_reshaped, mask)
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

    ###### STACKS ######

    def query_stack(self, q_in, mask, scope):
        '''
        Single query stack 
        Args:
            q_in: (batch_size, seqlen, embed_size)
            mask: (batch_size, seqlen, seqlen)
        '''
        with tf.variable_scope(scope):
            out = layer_norm(out + multihead_attention(q_in, q_in, q_in, mask))
            out = layer_norm(out + feed_forward(out))

        return out

    def passage_stack(self, p_in, q_out, input_mask, target_mask, scope):
        '''
        Single passage stack
        Args:
            p_in: (batch_size, seqlen, embed_size)
            q_out: output from query stack
        '''
        with tf.variable_scope(scope):
            out = layer_norm(p_in + multihead_attention(p_in, p_in, p_in, mask = target_mask))
            out = layer_norm(out + multihead_attention(out, out, q_out, mask = input_mask))
            out = layer_norm(out + feed_forward(out))

        return out

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

    '''
    MODEL FUNCTIONS
    ===============

    Following functions are used for operation for the model
    '''

    def make_tf_iterators(self, q, p, l):
        '''
        since loading via tensorflows build-in functions can significantly boost speed,
        trying to make some here.
        '''
        pass

    def make_basic_iterators(self, q, p, l):


    def train(self, queries_, passage_, label_, num_epochs = 50):
        '''
        This is the function used to train the model.
        Args:
            queries_: numpy array for queries
            passage_: numpy array for passages
            label_: numpy array for labels
        '''
        # checks
        assert len(queries_) == len(passage_) == len(label_)

        if not self.is_training:
            raise ValueError("Config not up for training,", self.is_training)

        for ep in range(num_epochs):
            # for each epoch, go over the entire dataset once










        
