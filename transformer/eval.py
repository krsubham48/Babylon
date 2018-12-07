'''
Evaluation file for various models.

Aim of this file is to be as generic as possible so training any model is a breeze
all the major functions are already present here and the only thing that should can
be changed is the network file.
'''

# dependencies
import argparse
import numpy as np
import os

# custom model
import network

def load_numpy_array(filepath):
	return np.load(filepath)

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
	parser.add_argument('--eval-file', type = str, help = 'path to evaluation file')
    parser.add_argument('--embedding-path', type = str, help = 'path to embedding .npy file')
    parser.add_argument('--all-words', type = str, help = 'path to words.txt file')
    parser.add_argument('--query-ids', type = str, help = 'path to list of IDs for query and passages')
    parser.add_argument('--model-path', type = str, help = 'path to folder where saving model')
    parser.add_argument('--results-path', type = str, help = 'path to folder where saving results')
    args = parser.parse_args()

    '''
	Step 1: Before the models is built and setup, do the preliminary work
    '''

    # modes
    is_training = False # we are training the model here

    # make the folders
	if not os.path.exists(args.model_folder):
		os.makedirs(args.model_folder)
		print('[!] Model saving folder could not be found, making folder, {args.model_folder}')

	'''
    Step 2: All the checks are done, make the model
    '''

    # load the training numpy matrix
    train_q = load_numpy_array(args.q_file)
    train_p = load_numpy_array(args.p_file)
    train_l = load_numpy_array(args.l_file)
    embedding_matrix = load_numpy_array(args.emb_file)

    # load the model, this is one line that will be changed for each case
    model = network.TransformerNetwork(scope = args.model_name,
                                       save_folder = args.save_folder,
                                       pad_id = pad_id,,
                                       is_training = is_training,
                                       dim_model = 50,
                                       ff_mid = 128,
                                       ff_mid1 = 128,
                                       ff_mid2 = 128,
                                       num_stacks = 2,
                                       num_heads = 5)

    '''
    Step 3: Train the model
    '''

    # train the model
    model.eva(queries_ = train_q,
              passage_ = train_p,
              query_ids = args.query_ids,
              output_file = args.results_path)

