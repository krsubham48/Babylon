# Transformer Network

This model is inspired from the [transformer network](https://arxiv.org/abs/1706.03762) from 'Attention is all you need'. There are some major differences between our model and the model in the paper (for reading material goto `/readings` folder).

## Run Model

\[Development\] To run the model you need to have the data in the required format. Read `../common/` to see how get data in required format. The model needs to be first trained, to train it run the command

```bash
$ python3 train.py --args
```

To run the trained model run the command:

```bash
$ python3 eval.py --args
```

## Files

The transformer network has the following files:

1. `network.py`: Network architecture class, this only holds the crucial functions. As many as possible functions have been pushed to seperate files.

2. `utils.py`: Util functions and DataManager class for easier handling of training data.

3. `train.py`: File to train the model using training files. It has following arguments: [Development]

```
--qpl-file:       path to numpy dumps
--emb-file:       path to ambedding matrix
--save-folder:    path to folder where saving model
--num-epochs:     number fo epochs for training
--model-name:     name of the model
--val-split:      validation split ratio
--ft-factor:      ratio of false samples to true samples (should be <= 9)
--save-frequency: save model after these steps
--seqlen:         lenght of longest sequence
--batch-size:     size of minibatch
--thresh-upper:   upper threshold for dummy accuracy check
--thresh-lower:   lower threshold for dummy accuracy check
--jitter:         helps in dummy accuracy calculation
```

4. `eval.py`: File to run the model for obtaining results on evaluation file. It has following arguments: [Development]

```
--eval-file:      path to evaluation file
--embedding-path: path to embedding .npy file
--all-words:      path to words.txt file
--query-ids:      path to list of IDs for query and passages
--model-path:     path to folder where saving model
--results-path:   path to folder where saving results
```

5. `net_test.py`: Test file to check functionality of network file, dumps all the information in terminal. It has following arguments:

```
--num-examples: number of examples
--qp-factor:    num passages per query
--vocab-size:   size of vocabulary
--emb-dim:      embedding dimension
--seqlen:       length of sequences
--save-folder:  path to folder where saving model
```

6. REDUNDANT, `model_config.py`: Configuration values to run the model and operate it end to end. Moved all the values to the network class making handling easier.

7. REDUNDANT, `core_layers.py`: All the network layers to build the model. Moved all the functions into the network class, was becoming uselessly complicated.

There also are some jupyter notebooks in the `/notebooks` folder. They are merely used for quick prototyping and not the actual development of the model.

## Results

### Download the model

You can download the model from this [folder]() `NOTHING RIGHT NOW`

## Update Log

All the dates are given in `DD-MM-YYYY` format.

07-12-2018: functionality for training model complete

21-11-2018: The first commit of the model, with basic skeleton code.