'''
File to load the sentences and convert them to numpy dumps. The idea is that we
can directly load these numpy dumps for training or testing. This is the final
file that we need to call to get the data converted to the dump.

Input Sentence: 'This sentence for this query is irrelevant'
Output array: [<PAD>, <PAD> ... (seqlen - input_len), 32, 42, 1, 32, 54, 909 ]

'''

# importing the dependencies

import argparse
import numpy as np
from sklearn.utils import shuffle

STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
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
    word2idx = dict((w, i) for i,w in enumerate(words))
    return word2idx

def cvt_srt2id(inp, word2idx, min_seqlen, max_seqlen):
    # convert string to list of ID
    words = re.split('\W', inp)
    words = [w for w in words if w and w not in STOPWORDS]
    sent_embd = []
    if min_seqlen <= len(words):
        if len(words) > max_seqlen + 1:
            words = words[:max_seqlen]
        for w in words:
            if w not in word2idx:
                w = '<UNK_{0}>'.format(str(np.random.randint(num_unk+1)))
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
    
    # shuffling is only done when in training mode, there is no need to do it in evaluation mode
    # otherwise we will have to store the hashing as well
    parser.add_argument('--shuffle', type = bool, default = True, help = 'to shuffle the data')

    parser.add_argument('--num-unk', type = str, default = 30, help = 'number of unique tokens')
    parser.add_argument('--buffer-size', type = int, default = 1000000, help = 'size of each buffer')
    parser.add_argument('--max-qlen', type = int, default = 12, help = 'maximum query length')
    parser.add_argument('--min-qlen', type = int, default = 2, help = 'minimum query length')
    parser.add_argument('--max-plen', type = int, default = 80, help = 'maximum passage length')
    parser.add_argument('--min-plen', type = int, default = 10, help = 'minimum passage length')
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

    while True:
        num_line += 1

        line = file.readline()
        if not line:
            # if last then dump
            save_data(args.training_mode, query_buffer, passages_buffer, labels_buffer, args.shuffle)
            print("...this was the last dump, exiting from the loops now")
            break

        # If not the last line then proceed
        tokens = line.split(' ').lower().split('\t')
        query, passage = tokens[1], tokens[2]
        if not args.training:
            label = tokens[3]

        # convert strings to ID
        query2ids = cvt_srt2id(query, word2idx, args.min_qlen, args.max_qlen)
        passage2ids = cvt_srt2id(passage, word2idx, args.min_plen, args.max_plen)

        # add to data if conditions in cvt_srt2id satisfied
        if query2ids and passage2ids:
            query_buffer.append(query2ids)
            passages_buffer.append(passage2ids)
            if not args.training:
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