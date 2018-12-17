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
import network # network
from utils import load_numpy_array # numpy loading

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qpl-file', type = str, help = 'path to numpy dumps')
    parser.add_argument('--emb-file', type = str, help = 'path to ambedding matrix')
    parser.add_argument('--save-folder', type = str, default = './saved', help = 'path to folder where saving model')
    parser.add_argument('--num-epochs', type = int, default = 5, help = 'number fo epochs for training')
    parser.add_argument('--model-name', type = str, default = 'Babylon', help = 'name of the model')
    parser.add_argument('--val-split', type = float, default = 0.1, help = 'validation split ratio')
    parser.add_argument('--save-frequency', type = int, default = 5000, help = 'save model after these steps')
    parser.add_argument('--seqlen', type = int, default = 80, help = 'lenght of longest sequence')
    parser.add_argument('--batch-size', type = int, default = 1024, help = 'size of minibatch')
    parser.add_argument('--thresh-upper', type = float, default = 0.9, help = 'upper threshold for dummy accuracy check')
    parser.add_argument('--thresh-lower', type = float, default = 0.2, help = 'lower threshold for dummy accuracy check')
    args = parser.parse_args()

    '''
    Step 1: Before the models is built and setup, do the preliminary work
    '''

    # make the folders
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        print('[*] Model saving folder could not be found, making folder ', args.save_folder)

    # We need to get the list of all the q, p and l files that were generated
    q_paths = sorted(glob(args.qpl_file + '_q*.npy'))
    p_paths = sorted(glob(args.qpl_file + '_p*.npy'))
    l_paths = sorted(glob(args.qpl_file + '_l*.npy'))

    print(q_paths)
    print(p_paths)
    print(l_paths)

    '''
    Step 2: All the checks are done, make the model
    '''
    print('[*] Data loading ...')
    # load the training numpy matrix
    for i in range(len(q_paths)):
        print('... loading file number:', i)
        if i == 0:
            train_q = load_numpy_array(q_paths[i])
            train_p = load_numpy_array(p_paths[i])
            train_l = load_numpy_array(l_paths[i])
        else:
            q_ = load_numpy_array(q_paths[i])
            p_ = load_numpy_array(p_paths[i])
            l_ = load_numpy_array(l_paths[i])
            train_q = np.concatenate([train_q, q_])
            train_p = np.concatenate([train_p, p_])
            train_l = np.concatenate([train_l, l_])
    
    # load embedding matrix
    print('... loading embedding matrix')
    embedding_matrix = load_numpy_array(args.emb_file)

    print('[*] ... Data loading complete!')

    # load the model, this is one line that will be changed for each case
    print('[*] Making model')
    model = network.TransformerNetwork(scope = args.model_name,
                                       save_folder = args.save_folder,
                                       pad_id = len(embedding_matrix),
                                       save_freq = args.save_frequency,
                                       is_training = True,
                                       dim_model = embedding_matrix.shape[-1],
                                       ff_mid = 128,
                                       ff_mid1 = 128,
                                       ff_mid2 = 128,
                                       num_stacks = 2,
                                       num_heads = 5)

    # build the model
    print('[*] Building model (for details look at the stack below)')
    model.build_model(emb = embedding_matrix,
                      seqlen = args.seqlen,
                      batch_size = args.batch_size,
                      print_stack = True)

    '''
    Step 3: Train the model
    '''

    # train the model
    print('#### Training Model ####')
    model.train(queries_ = train_q,
                passage_ = train_p,
                label_ = train_l,
                num_epochs = args.num_epochs,
                val_split = args.val_split,
                smooth_thresh_upper = args.thresh_upper,
                smooth_thresh_lower = args.thresh_lower)

