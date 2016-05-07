"""
Preprocesses the story corpus
"""
import csv
import os, json
import numpy as np
import unicodedata
import re
from gensim.models.word2vec import Word2Vec
import time
import sys

word_idx_map = {'<NULL>': 0, '<UNK>': 1, '<END>': 2}

idx_word_map = ['<NULL>', '<UNK>', '<END>']

# we replace rare words with <UNK>, which shares the same vector
word_count_map = {}  # length: 34044

max_seq_len = 0

W_embed = None


def load_dataset(base_dir):
    data = {}

    train_file = os.path.join(base_dir, 'ROCStories__spring2016 - ROC-Stories-naacl-camera-ready.csv')

    get_train_data('train', train_file, data)

    val_file = os.path.join(base_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv')

    get_valid_data('val', val_file, data)

    test_file = os.path.join(base_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv')

    get_valid_data('test', test_file, data)

    return data


def get_train_data(category, file_path, data):
    """
    Args:
        category: 'train', 'dev', or 'test'
        data: pass in the dictionary, and we fill it up inside this function
    """
    global max_seq_len

    data[category + '_src_sentences'] = []
    data[category + '_tgt_sentences'] = []
    data['y_' + category] = []

    with open(file_path, 'rU') as f:
        reader = csv.DictReader(f, dialect="excel")
        # train_keys: ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        for row in reader:
            sentence1 = row['sentence1']
            sentence2 = row['sentence2']

            sentence1_array = sentence1.split()
            sentence2_array = sentence2.split()

            for word in sentence1_array:
                if word not in word_idx_map:
                    word_idx_map[word] = len(word_idx_map)
                    idx_word_map.append(word)
                if word not in word_count_map:
                    word_count_map[word] = 0
                else:
                    word_count_map[word] += 1

            for word in sentence2_array:
                if word not in word_idx_map:
                    word_idx_map[word] = len(word_idx_map)
                    idx_word_map.append(word)
                if word not in word_count_map:
                    word_count_map[word] = 0
                else:
                    word_count_map[word] += 1

            if len(sentence1_array) > max_seq_len:
                max_seq_len = len(sentence1_array)
            elif len(sentence2_array) > max_seq_len:
                max_seq_len = len(sentence2_array)

            # data['y_' + category].append(int(label_idx_map[json_obj['gold_label']]))
            data[category + '_src_sentences'].append(sentence1)
            data[category + '_tgt_sentences'].append(sentence2)

    data['y_' + category] = np.asarray(data['y_' + category], dtype='int32')

def get_valid_data(category, file_path, data):
    # valid/test keys: ['InputStoryid', 'InputSentence1', 'InputSentence2', 'InputSentence3',
    #           'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']

    pass

def convert_words_to_idx(data_X):
    """
    We convert word sentence into idx sentence,
    and if a word is not in word2vec: "rare", we already have a randomized word embedding

    Args:
        data_X: the 'train_sentences', 'dev_sentences', 'test_sentences'

    Returns:
    """

    converted = np.zeros((len(data_X), max_seq_len), dtype='int32')

    for i, sen in enumerate(data_X):
        sen_idx = np.zeros(max_seq_len, dtype='int32')

        for j, word in enumerate(sen.split()):
            sen_idx[j] = word_idx_map[word]

        sen_idx[len(sen)] = word_idx_map['<END>']  # append <END> token to it

        converted[i, :] = sen_idx

    return converted


def compress_word2vec(W_embed, model):
    """
    We compress word2vec's 1.5G file with
    only the words we have

    update W_embed

    word2vec: the word2vec model we loaded

    Returns:
    """

    num_words_not_in = 0

    for i, word in enumerate(idx_word_map):
        if word in model:
            W_embed[i, :] = model[word]
        else:
            num_words_not_in += 1

    print "words not in word2vec: ", num_words_not_in


if __name__ == '__main__':
    load_dataset('/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/allen/trident/data/story_corpus_16')
