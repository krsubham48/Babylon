# Common

These are all the functions and utilities that are common to all the models. 

## Running

What we are doing here is that we first convert the given glove embeddings to numpy dump and obtain the word list from it. Next we take the text and convert them into the format given in Output below. The following are the commands that should be run to get the final data:

```bash
$ python3 glove2npy --file-path=path/to/glove.txt --output-name=path/to/name --num-unk=20
$ python3 get_final_data.py --training=False --text-file=path/to/train.txt --output-name=path/to/output --num-unk=20
```

NOTE: `--num-unk` in both the files should be same, otherwise can cause error or give bad results

## Output

The output of the opration above is generation of multiple `.npy` dumps. The structure of each dump is as follows:

```
Query: [12, 33, 123, 43] x 10, [122, 95, 932, 532, 2234, 23] x 10 ... ]
Passages: [123, 95, 42, 90, 9923, 934, 853, 1245, 1239], ... ]
labels: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ... ]
```

## File Documentation

This folder has following files: (RED) means redundant.

1. `glove2npy.py`: The file to convert the given glove files to numpy format `*.npy` file. Following is the list of arguments:

```
--file-path:    path to training file
--output-name:  data file is output_name.npy, word file as output_name_words.txt
--num-unk:      number of unique word tokens
```

2. `get_final_data.py`: The file to get the final data. Following is the list of arguments:

```
--training-mode: training mode
--input-file:   path to text file
--output-name:  output is output_name_xx.npy
--words:        path to words.txt file
--shuffle:      to shuffle the data
--num-unk:      number of unique tokens
--buffer-size:  size of each buffer
--max-len:      maximum length
--min-len:      minimum length
```

3. `get_eval_data.py`: The file to get teh evaluation data dump. Following is the list of arguments:

```
--input-file:  path to text file
--output-name: output is output_name_xx.npy
--words:       path to words.txt file
--num-unk:     number of unique tokens
--buffer-size: size of each buffer
--max-len:     maximum length
```

4. `split_data.py`: (RED) The file to split and dump the data in smaller files of input number of lines. Following is the list of arguments:

```
--file-path:      path to training file
--output-name:    prefix for output file name, training file is output_name_train.txt
--num-sentences:  number of sentences in output training file
--ratio:          split ratio (val/total)
--randomize:      to randomise data. if True random points are selected
```

5. `tex2ctf_mod.py`: (RED) The file to dump the data in Microsoft CNTK format. This is the modified version of file in `/baselines`. Following is the list of arguments:

```
--mode:           operation mode, FULL for complete dump, SAMPLE for first 3000 lines
--train-file:     path to training file
--valid-file:     path to validation file
--eval-file:      path to evaluation file
--glove-file:     path to glove emdedding file
--max-query-len:  maximum length of query to be processed
--max-pass-len:   maximum length of passage to be processed
--prefix:         prefix for this dump iteration
--verbose:        verbosity, (True for yes)
```