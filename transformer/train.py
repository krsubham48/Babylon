'''
Training file for various models.

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
	parser.add_argument('--p-file', type = str, help = 'path to passage numpy dump')
	parser.add_argument('--q-file', type = str, help = 'path to query numpy dump')
	parser.add_argument('--l-file', type = str, help = 'path to labels numpy dump')
	parser.add_argument('--emb-file', type = str, help = 'path to ambedding matrix')
    parser.add_argument('--save-folder', type = str, help = 'path to folder where saving model')
    parser.add_argument('--model-name', type = str, default = 'Babylon', help = 'name of the model')
    parser.add_argument('--val-split', type = float, default = 0.1, help = 'validation split ratio')
    parser.add_argument('--save-frequency', type = int, default = 10, help = 'save model after these steps')
    args = parser.parse_args()

    '''
	Step 1: Before the models is built and setup, do the preliminary work
    '''

    # modes
    is_training = True # we are training the model here

    # make the folders
	if not os.path.exists(args.model_folder):
		os.makedirs(args.model_folder)
		print('[!] Model saving folder could not be found, making folder ', args.model_folder)

	# load training sentences and

	'''
	Step 2: All the defaults are done now, everything is ready to make the model
	'''
	# load the training numpy matrix
	train_p = load_numpy_array(args.p_file)
	train_q = load_numpy_array(args.q_file)
	train_l = load_numpy_array(args.l_file)
	embedding_matrix = load_numpy_array(args.emb_file)

	# load the model, this is one line that will be changed for each case
	model = network.TransformerNetwork(args.model_name)

	# add attributes to the model
	model.save_freq = args.save_frequency
	model.save_folder = args.model_folder

	model.build_model(word_embeddings, ...)

	# choosing to train inside the class itself
	# model.train(X, Y, word2idx, verbosity, ...) 



