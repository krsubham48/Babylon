'''
File to load the sentences and convert them to numpy dumps. The idea is that we
can directly load these numpy dumps for training or testing. This is the final
file that we need to call to get the data converted to the dump.

Input Sentence: 'This sentence for this query is irrelevant'
Output array: [<PAD>, <PAD> ... (seqlen - input_len), 32, 42, 1, 32, 54, 909 ]

'''

# importing the dependencies

import re # Regex
import argparse # parsing arguments
import numpy as np # linear algebra
from sklearn.utils import shuffle # shuffling data

'''
NOTE: removed question words - what, why, who, where
'''

STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
            'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
            'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
            "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def save_npy(filepath, arr):
    # save numpy array
    np.save(filepath, arr)

def get_word2idx(filepath):
    # get word2idx
    f = open(filepath)
    words = f.readlines()
    word2idx = dict((w.split('\n')[0], i) for i,w in enumerate(words))
    # print(len(word2idx))
    # print(word2idx['<UNK_9>'])
    return word2idx

def cvt_srt2id(inp, word2idx, min_seqlen, max_seqlen, num_unk):
    # convert string to list of ID
    # print(inp)
    words = re.split('\W', ' '.join(inp))
    words = [w for w in words if w and w not in STOPWORDS]
    sent_embd = []
    if min_seqlen <= len(words):
        if len(words) > max_seqlen + 1:
            words = words[:max_seqlen]
        for w in words:
            if w not in word2idx:
                w = '<UNK_{0}>'.format(str(np.random.randint(int(num_unk))))
            embd = word2idx[w]
            sent_embd.append(embd)

        return sent_embd

    return None

def save_data(trn, q_buf, p_buf, l_buf = None, shfl = True):
    if trn:
        # shfl and trn or trn = shfl
        print("[!] Performing shuffling of data... (this may take some time)")
        
        # convert to numpy arrays
        q_buf = np.array(q_buf)
        p_buf = np.array(p_buf)
        l_buf = np.array(l_buf)

        if shfl:
            # shuffle the values
            q_buf, p_buf, l_buf = shuffle(q_buf, p_buf, l_buf)

        q_path = args.output_name + '_q{0}.npy'.format(num_buffer)
        print("[*]Saving file...", q_path)
        save_npy(q_path, q_buf)

        p_path = args.output_name + '_p{0}.npy'.format(num_buffer)
        print("[*]Saving file...", p_path)
        save_npy(p_path, p_buf)

        l_path = args.output_name + '_l{0}.npy'.format(num_buffer)
        print("[*]Saving file...", l_path)
        save_npy(l_path, l_buf)

    else:
        print("[!] Performing shuffling of data... (this may take some time)")
        
        # convert to numpy arrays
        q_buf = np.array(q_buf)
        p_buf = np.array(p_buf)

        if shlf:
            # shuffle the values
            q_buf, p_buf = shuffle(q_buf, p_buf)

        q_path = args.output_name + '_q{0}.npy'.format(num_buffer)
        print("[*]Saving file...", q_path)
        save_npy(q_path, q_buf)

        p_path = args.output_name + '_p{0}.npy'.format(num_buffer)
        print("[*]Saving file...", p_path)
        save_npy(p_path, p_buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-mode', type = bool, default = False, help = 'training mode')
    parser.add_argument('--input-file', type = str, help = 'path to text file')
    parser.add_argument('--output-name', type = str, default = './dump', help = 'output is output_name_xx.npy')
    parser.add_argument('--words', type = str, help = 'path to words.txt file')
    
    # shuffling is only done when in training mode, there is no need to do it in evaluation mode
    # otherwise we will have to store the hashing as well
    parser.add_argument('--shuffle', type = bool, default = True, help = 'to shuffle the data')

    parser.add_argument('--num-unk', type = str, default = 30, help = 'number of unique tokens')
    parser.add_argument('--buffer-size', type = int, default = 1000000, help = 'size of each buffer')

    # sequence length is same for both
    parser.add_argument('--max-len', type = int, default = 12, help = 'maximum length')
    parser.add_argument('--min-len', type = int, default = 2, help = 'minimum length')
    args = parser.parse_args()

    '''
    load the text and preprocess it
    '''
    file = open(args.input_file)

    query_buffer = []
    passages_buffer = []
    labels_buffer = []

    # number of lines processed
    num_line = 0
    num_buffer = 0

    # get word2idx
    print('[*] Getting word2idx...')
    word2idx = get_word2idx(args.words)

    print('[#] len word2idx:', len(word2idx))

    while True:
        num_line += 1

        line = file.readline()
        if not line:
            # if last then dump
            save_data(args.training_mode, query_buffer, passages_buffer, labels_buffer, args.shuffle)
            print("...this was the last dump, exiting from the loops now")
            break

        # print(line)

        # If not the last line then proceed
        tokens = line.split('\t')
        # print(tokens)
        tokens = [t.lower() for t in tokens]
        query, passage = tokens[1].split(' '), tokens[2].split(' ')
        if args.training_mode:
            label = tokens[3]

        # convert strings to ID
        query2ids = cvt_srt2id(query, word2idx, args.min_len, args.max_len, args.num_unk)
        passage2ids = cvt_srt2id(passage, word2idx, args.min_len, args.max_len, args.num_unk)

        # add to data if conditions in cvt_srt2id satisfied
        if query2ids and passage2ids:
            query_buffer.append(query2ids)
            passages_buffer.append(passage2ids)
            if args.training_mode:
                labels_buffer.append(float(label))

        # if length of buffers exceeds given buffer size
        if len(query_buffer) == args.buffer_size:

            save_data(args.training_mode, query_buffer, passages_buffer, labels_buffer, args.shuffle)

            # reset buffer
            query_buffer = []
            passages_buffer = []
            labels_buffer = []
            num_buffer += 1

    print("[*]... execution completed, Exiting!")