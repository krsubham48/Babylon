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
        if len(words) < max_seqlen + 1:
            words = words[:max_seqlen]
        for w in words:
            if w not in word2idx:
                w = '<UNK_{0}>'.format(str(np.random.randint(num_unk)))
            embd = word2idx[w]
            sent_embd.append(embd)

        return sent_embd
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type = bool, default = False, help = 'training mode')
    parser.add_argument('--text-file', type = str, help = 'path to text file')
    parser.add_argument('--output-name', type = str, default = './dump', help = 'output is output_name_xx.npy')
    parser.add_argument('--num-unk', type = str, help = 'number of unique tokens')
    parser.add_argument('--buffer-size', type = int, default = 1000000, help = 'size of each buffer')
    parser.add_argument('--max-querylen', type = int, default = 12, help = 'maximum query length')
    parser.add_argument('--min-querylen', type = int, default = 2, help = 'minimum query length')
    parser.add_argument('--max-passlen', type = int, default = 80, help = 'maximum passage length')
    parser.add_argument('--min-passlen', type = int, default = 10, help = 'minimum passage length')
    args = parser.parse_args()

    '''
    load the text and prerocess it
    '''
    file = open(args.text_file)

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
            save_npy(args.output_name + '_q{0}.npy'.format(num_buffer), np.array(query_buffer))
            save_npy(args.output_name + '_p{0}.npy'.format(num_buffer), np.array(passages_buffer))
            if not args.training:
                save_npy(args.output_name + '_l{0}.npy'.format(num_buffer), np.array(labels_buffer))
            break

        tokens = line.split(' ').lower().split('\t')
        query, passage = tokens[1], tokens[2]
        if not args.training:
            label = tokens[3]

        query2ids = cvt_srt2id(query, word2idx, args.min_querylen, args.max_querylen)
        passage2ids = cvt_srt2id(passage, word2idx, args.min_passlen, args.max_passlen)
        if query and passage:
            query_buffer.append(query2ids)
            passages_buffer.append(passage2ids)
            if not args.training:
                labels_buffer.append(float(label))

        # if length of buffers exceed
        if len(query_buffer) == args.buffer_size:
            save_npy(args.output_name + '_q{0}.npy'.format(num_buffer), np.array(query_buffer))
            save_npy(args.output_name + '_p{0}.npy'.format(num_buffer), np.array(passages_buffer))
            if not args.training:
                save_npy(args.output_name + '_l{0}.npy'.format(num_buffer), np.array(labels_buffer))

            # reset buffer
            query_buffer = []
            passages_buffer = []
            labels_buffer = []
            num_buffer += 1
