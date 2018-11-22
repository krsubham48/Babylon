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
	Step 2: All the defaults are done now, everything is ready to make the model
	'''
	word2idx = get_wordIds(args.all_words)
	word_embeddings = load_numpy_array(args.embedding_path)

	# load the model, this is one line that will be changed for each case
	model = network.TransformerNetwork(args.model_name, is_training)
	model.load_model(args.model_path)

	# model.initialize_embeddings(word_embeddings)
	# model.eval_score(args.)