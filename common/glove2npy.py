'''
Script to convert the input embeddings to numpy array dumps.

Automatically add <PAD> and <UNK_> tokens
'''

import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type = str, help = 'path to training file')
    parser.add_argument('--output-name', type = str,
        help = 'data file is output_name.npy, word file as output_name_words.txt')
    parser.add_argument('--num-unk', type = int, help = 'number of unique word tokens')
    args = parser.parse_args()

    # open the file
    f = open(args.file_path, 'r')

    # buffer matrices
    embeddings_ = []
    all_words = []

    num_lines_processed = 0

    while True:
        num_lines_processed += 1
        line = f.readline()
        if not line:
            break

        tokens = line.split('\n')[0].split(' ')
        word = tokens[0]
        embd = np.array([np.float32(i) for i in tokens[1:]])
        all_words.append(str(word))
        embeddings_.append(embd)

        # print(line_WE)

        '''
        if len(embeddings_) == 0:
            embeddings_ = embd

        else:
            embeddings_ = np.append(embeddings_, embd)
        '''

    # add unique tokens
    for i in range(args.num_unk):
        all_words.append('<UNK_{0}>'.format(i))
        embd = 2*np.random.randn(embd.shape[0]) - 1
        embeddings_.append(embd)

    # add padding
    all_words.append('<PAD>')
    embd = 2*np.zeros(embd.shape[0], dtype = np.float32)
    embeddings_.append(embd)

    # save the data 
    print('Number of buffers handled:', num_lines_processed)
    embeddings_ = np.array(embeddings_)

    print('[*] Shape of the final embedding matrix:',embeddings_.shape)
    print('[!] Saving matrix at:', args.output_name + '.npy')

    np.save(args.output_name + '.npy', embeddings_)
    del embeddings_
    
    # save the words
    words = '\n'.join(all_words)
    print('[!] Saving words at:', args.output_name + '_words.txt')

    f = open(args.output_name + '_words.txt', 'w')
    f.write(words)
    f.close()
