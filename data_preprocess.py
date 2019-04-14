import operator
import os

import numpy as np
import pandas as pd
from sklearn import model_selection

from utils import *


def dict_words():
    # Make a list of the folders in the dataset
    directory = [f for f in os.listdir('./20_newsgroups') if not f.startswith('.')]
    # Create a dictionary of words with their frequency
    vocab = {}
    for i in range(len(directory)):
        # Create a list of files in the given dictionary
        files = os.listdir('./20_newsgroups/' + directory[i])

        for j in range(len(files)):
            # Path of each file
            path = './20_newsgroups/' + directory[i] + '/' + files[j]

            # open the file and read it
            text = open(path, 'r', errors='ignore').read()

            for word in text.split():
                if len(word) != 1:
                    # Check if word is a non stop word or non block word(we have created) only then proceed
                    if not word.lower() in stop_words:
                        if not word.lower() in block_words:
                            # If word is already in dictionary then we just increment its frequency by 1
                            if vocab.get(word.lower()) != None:
                                vocab[word.lower()] += 1

                            # If word is not in dictionary then we put that word in our dictinary by making its frequnecy 1
                            else:
                                vocab[word.lower()] = 1

    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

    # Dictionary containing the most occuring k-words.
    kvocab = {}

    # Frequency of 1000th most occured word
    z = sorted_vocab[2000][1]

    for x in sorted_vocab:
        kvocab[x[0]] = x[1]

        if x[1] <= z:
            break

    features_list = list(kvocab.keys())

    # Create a Dataframe containing features_list as columns
    df = pd.DataFrame(columns=features_list)

    # Filling the x_train values in dataframe
    for i in range(len(directory)):
        # Create a list of files in the given dictionary
        files = os.listdir('./20_newsgroups/' + directory[i])

        for j in range(len(files)):
            # Insert a row at the end of Dataframe with all zeros
            df.loc[len(df)] = np.zeros(len(features_list))

            # Path of each file
            path = './20_newsgroups/' + directory[i] + '/' + files[j]

            # open the file and read it
            text = open(path, 'r', errors='ignore').read()

            for word in text.split():
                if word.lower() in features_list:
                    df[word.lower()][len(df) - 1] += 1

    # Making the 2d array of x
    x = df.values

    # Feature list
    f_list = list(df)

    # Creating  y array containing labels for classification
    y = []
    for i in range(len(directory)):
        # Create a list of files in the given dictionary
        files = os.listdir('./20_newsgroups/' + directory[i])

        for j in range(len(files)):
            y.append(i)

    y = np.array(y)

    # Splitting the whole dataset for training and testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test
