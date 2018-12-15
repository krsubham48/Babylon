'''
Training file for various models.

Aim of this file is to be as generic as possible so training any model is a breeze
all the major functions are already present here and the only thing that should can
be changed is the network file.
'''

# dependencies
import argparse # argparse
import numpy as np # linear algebra
import os # OS 
from glob import glob # file handling

# custom model
import network

def get_wordIds(filepath):
    '''
    read the text file to get all the words
    convert to dictionary we do not need inverse dictionary
    since we are not predicting words but merely similarity
    '''
    f = open(filepath)
    words = f.readlines()
    word2idx = dict((w, idx) for idx, w in enumerate(words))

    f.close()
    del words

    return word2idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qpl-file', type = str, help = 'path to numpy dumps')
    parser.add_argument('--emb-file', type = str, help = 'path to ambedding matrix')
    parser.add_argument('--save-folder', type = str, default = './saved', help = 'path to folder where saving model')
    parser.add_argument('--num-epochs', type = int, default = 50, help = 'number fo epochs for training')
    parser.add_argument('--model-name', type = str, default = 'Babylon', help = 'name of the model')
    parser.add_argument('--val-split', type = float, default = 0.1, help = 'validation split ratio')
    parser.add_argument('--save-frequency', type = int, default = 10, help = 'save model after these steps')
    args = parser.parse_args()

    '''
    Step 1: Before the models is built and setup, do the preliminary work
    '''

    # make the folders
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        print('[*] Model saving folder could not be found, making folder ', args.save_folder)

    # We need to get the list of all the q, p and l files that were generated
    q_paths = sort(glob(args.qpl_file + '_q*'))
    p_paths = sort(glob(args.qpl_file + '_p*'))
    l_paths = sort(glob(args.qpl_file + '_l*'))

    '''
    Step 2: All the checks are done, make the model
    '''

    # load the training numpy matrix
    for i in range(len(q_paths)):
        if i == 0
            train_q = load_numpy_array(q_paths[i])
            train_p = load_numpy_array(p_paths[i])
            train_l = load_numpy_array(l_paths[i])
        else:
            train_q = np.stack([train_q, load_numpy_array(q_paths[i])])
            train_p = np.stack([train_p, load_numpy_array(p_paths[i])])
            train_l = np.stack([train_l, load_numpy_array(l_paths[i])])

    # reshape the matrices
    train_q = np.reshape(train_q, [-1])
    train_p = np.reshape(train_q, [-1])
    train_l = np.reshape(train_l, [-1])
    
    # load embedding matrix
    embedding_matrix = load_numpy_array(args.emb_file)

    # load the model, this is one line that will be changed for each case
    model = network.TransformerNetwork(scope = args.model_name,
                                       save_folder = args.save_folder,
                                       pad_id = len(embedding_matrix),
                                       save_freq = args.save_freq,
                                       is_training = True,
                                       dim_model = embedding_matrix.shape[-1],
                                       ff_mid = 128,
                                       ff_mid1 = 128,
                                       ff_mid2 = 128,
                                       num_stacks = 2,
                                       num_heads = 5)

    # build the model
    model.build_model(emb = embedding_matrix,
                      seqlen = seqlen,
                      print_stack = True)

    '''
    Step 3: Train the model
    '''

    # train the model
    model.train(queries_ = train_q,
                passage_ = train_p,
                label_ = train_l,
                num_epochs = args.num_epochs,
                val_split = args.val_split)

