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

word_idx_map = {'<NULL>': 0, '<UNK>': 1, '<END>': 2, '<START>': 3}

idx_word_map = ['<NULL>', '<UNK>', '<END>', '<START>']

# we replace rare words with <UNK>, which shares the same vector
word_count_map = {}  # length: 34044

# we collect max sentence length on all positions
max_seq_len = [0, 0, 0, 0, 0, 0]
# [20, 20, 20, 20, 20, 17]

max_src_seq_len = 0
# 65

W_embed = None


def load_dataset(base_dir):
    data = {}

    train_file = os.path.join(base_dir, 'ROCStories__spring2016 - ROC-Stories-naacl-camera-ready.csv')

    get_train_data('train', train_file, data)

    val_file = os.path.join(base_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv')

    get_valid_test_data('val', val_file, data)

    test_file = os.path.join(base_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv')

    get_valid_test_data('test', test_file, data)

    return data


def flatten(list_to_flatten):
    l = []
    for elem in list_to_flatten:
        if isinstance(elem, (list, tuple)):
            for x in flatten(elem):
                l.append(x)
        else:
            l.append(elem)
    return l


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC

    (this removes "." period as well)
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def get_train_data(category, file_path, data):
    """
    Args:
        category: 'train'
        data: pass in the dictionary, and we fill it up inside this function
    """

    global max_src_seq_len

    data[category + "_src_sentences_merged"] = []

    with open(file_path, 'rU') as f:
        reader = csv.DictReader(f, dialect="excel")
        # train_keys: ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        for row in reader:
            merged = []
            l1 = process_single_sentence(1, category, row, data, merged)
            l2 = process_single_sentence(2, category, row, data, merged)
            l3 = process_single_sentence(3, category, row, data, merged)
            l4 = process_single_sentence(4, category, row, data, merged)
            process_single_sentence(5, category, row, data)

            if l1 + l2 + l3 + l4 > max_src_seq_len:
                max_src_seq_len = l1 + l2 + l3 + l4

            data[category + "_src_sentences_merged"].append(flatten(merged))


def get_valid_test_data(category, file_path, data):
    """
    :param category: 'val', or 'test'
    :param file_path:
    :param data:
    :return:
    """
    global max_src_seq_len

    data[category + "_src_sentences_merged"] = []

    data['y_' + category] = []

    with open(file_path, 'rU') as f:
        reader = csv.DictReader(f, dialect="excel")
        # valid/test keys: ['InputStoryid', 'InputSentence1', 'InputSentence2', 'InputSentence3',
        #           'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']
        for row in reader:
            merged = []
            l1 = process_single_sentence_valid(1, category, row, data, merged)
            l2 = process_single_sentence_valid(2, category, row, data, merged)
            l3 = process_single_sentence_valid(3, category, row, data, merged)
            l4 = process_single_sentence_valid(4, category, row, data, merged)
            process_single_sentence_valid(5, category, row, data)
            process_single_sentence_valid(6, category, row, data)

            if l1 + l2 + l3 + l4 > max_src_seq_len:
                max_src_seq_len = l1 + l2 + l3 + l4

            data['y_' + category].append(int(row['AnswerRightEnding']))

            data[category + "_src_sentences_merged"].append(flatten(merged))

    data['y_' + category] = np.asarray(data['y_' + category], dtype='int32')


def process_single_sentence(sentence_num, category, row, data, merged=None):
    """
    :param sentence_num:
    :param sentence_num: [1,2,3,4,5] - 5 sentences' position, used to collect length
                for max sequence length. [5] is reserved for decoder
    :param decoder:
    :return:
    """
    if category + "_sentence" + str(sentence_num) not in data:
        data[category + "_sentence" + str(sentence_num)] = []

    sentence = clean_str(row['sentence' + str(sentence_num)])  # so punctuations can be gone
    sentence_array = sentence.split()

    for word in sentence_array:
        if word not in word_idx_map:
            word_idx_map[word] = len(word_idx_map)
            idx_word_map.append(word)
        if word not in word_count_map:
            word_count_map[word] = 0
        else:
            word_count_map[word] += 1

    if len(sentence_array) > max_seq_len[sentence_num - 1]:
        max_seq_len[sentence_num - 1] = len(sentence_array)

    data[category + "_sentence" + str(sentence_num)].append(sentence_array)

    if merged is not None:
        merged.append(sentence_array)

    return len(sentence_array)


def process_single_sentence_valid(sentence_num, category, row, data, merged=None):
    # valid/test keys: ['InputStoryid', 'InputSentence1', 'InputSentence2', 'InputSentence3',
    #           'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']

    if category + "_sentence" + str(sentence_num) not in data:
        data[category + "_sentence" + str(sentence_num)] = []

    if sentence_num > 4:
        label = 'RandomFifthSentenceQuiz' + str(sentence_num - 4)
    else:
        label = 'InputSentence' + str(sentence_num)

    sentence = clean_str(row[label])

    sentence_array = sentence.split()

    for word in sentence_array:
        if word not in word_idx_map:
            word_idx_map[word] = len(word_idx_map)
            idx_word_map.append(word)
        if word not in word_count_map:
            word_count_map[word] = 0
        else:
            word_count_map[word] += 1

    if len(sentence_array) > max_seq_len[sentence_num - 1]:
        max_seq_len[sentence_num - 1] = len(sentence_array)

    data[category + "_sentence" + str(sentence_num)].append(sentence_array)

    if merged is not None:
        merged.append(sentence_array)

    return len(sentence_array)


def convert_words_to_idx(data):
    """
    We convert word sentence into idx sentence,
    and if a word is not in word2vec: "rare", we already have a randomized word embedding

    Args:
        data:
        ['train_sentence1', 'val_sentence1', 'train_sentence3', 'train_sentence2', 'train_sentence5', 'train_sentence4',
        'val_sentence5', 'test_sentence6', 'test_sentence4', 'test_sentence5', 'test_sentence2', 'val_sentence2',
        'test_sentence1', 'val_sentence3', 'y_val', 'test_sentence3', 'val_sentence6', 'val_sentence4', 'y_test',
        'train_src_sentences_merged']

    Returns:
    """
    cates = ['train', 'val', 'test']
    for cate in cates:
        for sen_num in xrange(1, 7):  # iterating from 1 to 6
            if sen_num == 6 and cate == 'train':
                break
            dataX = data[cate + "_sentence" + str(sen_num)]
            converted = np.zeros((len(dataX), max_seq_len[sen_num - 1]), dtype='int32')

            for i, sen in enumerate(dataX):
                sen_idx = np.zeros(max_seq_len[sen_num - 1], dtype='int32')

                if cate == 'val' or cate == 'test':
                    sen_idx[0] = word_idx_map['<START>']

                for j, word in enumerate(sen):
                    sen_idx[j] = word_idx_map[word]

                sen_idx[len(sen)] = word_idx_map['<END>']  # append <END> token to it

                converted[i, :] = sen_idx

            data[cate + "_sentence" + str(sen_num)] = converted


def merge_convert_src_sentences(data):
    """
    We convert merged source sentences into one representation
    'train_src_sentences_merged'

    category: train or val or test
    :return:
    """
    cates = ['train', 'val', 'test']
    for cate in cates:
        merged_sens = data[cate + "_src_sentences_merged"]
        converted = np.zeros((len(merged_sens), max_src_seq_len), dtype='int32')
        for i, sen in enumerate(merged_sens):
            sen_idx = np.zeros(max_src_seq_len, dtype='int32')
            for j, word in enumerate(sen):
                sen_idx[j] = word_idx_map[word]

            sen_idx[len(sen)] = word_idx_map['<END>']

            converted[i, :] = sen_idx

        data[cate + "_src_sentences_merged"] = converted


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
    begin = time.time()

    pwd = os.path.dirname(os.path.realpath(__file__))

    data = load_dataset(pwd + '/story_corpus_16')
    # ['train_sentence1', 'val_sentence1', 'train_sentence3', 'train_sentence2', 'train_sentence5', 'train_sentence4',
    #  'val_sentence5', 'test_sentence6', 'test_sentence4', 'test_sentence5', 'test_sentence2', 'val_sentence2',
    #  'test_sentence1', 'val_sentence3', 'y_val', 'test_sentence3', 'val_sentence6', 'val_sentence4', 'y_test',
    #  'train_src_sentences_merged']

    max_seq_len = [i + 1 for i in max_seq_len]  # index starts at 0

    max_seq_len[4] += 1  # add a <START> symbol for decoding sentences
    max_seq_len[5] += 1

    max_src_seq_len += 1

    print "max sequence stats: ", max_seq_len
    print "longest plot line: ", max_src_seq_len
    print data.keys()

    print "unique words: ", len(idx_word_map)

    merge_convert_src_sentences(data)

    convert_words_to_idx(data)

    print "data loaded..."

    # model = Word2Vec.load_word2vec_format(pwd + '/GoogleNews-vectors-negative300.bin.gz', binary=True)

    W_embed = np.random.randn(len(idx_word_map), 300)

    W_embed[0, :] = np.zeros(300, dtype='float32')

    W_embed /= 100

    # compress_word2vec(W_embed, model)

    with open(pwd + '/story_vocab.json', 'w') as outfile:
        json.dump({'idx_word_map': idx_word_map, 'word_idx_map': word_idx_map}, outfile)

    # ['test_sentence6', 'test_sentence4', 'test_sentence5', 'test_sentence2', 'test_sentence3', 'test_sentence1',
    #  'val_src_sentences_merged', 'train_sentence1', 'train_sentence3', 'train_sentence2', 'train_sentence5',
    #  'train_sentence4', 'test_src_sentences_merged', 'val_sentence1', 'val_sentence2', 'val_sentence3', 'val_sentence4',
    #  'val_sentence5', 'val_sentence6', 'y_test', 'y_val', 'train_src_sentences_merged']

    np.savez_compressed(pwd + "/snli_processed", W_embed=W_embed,
                        train_sentence1=data['train_sentence1'],
                        train_sentence2=data['train_sentence2'],
                        train_sentence3=data['train_sentence3'],
                        train_sentence4=data['train_sentence4'],
                        train_sentence5=data['train_sentence5'],
                        val_sentence1=data['val_sentence1'],
                        val_sentence2=data['val_sentence2'],
                        val_sentence3=data['val_sentence3'],
                        val_sentence4=data['val_sentence4'],
                        val_sentence5=data['val_sentence5'],
                        val_sentence6=data['val_sentence6'],
                        test_sentence1=data['test_sentence1'],
                        test_sentence2=data['test_sentence2'],
                        test_sentence3=data['test_sentence3'],
                        test_sentence4=data['test_sentence4'],
                        test_sentence5=data['test_sentence5'],
                        test_sentence6=data['test_sentence6'],
                        train_src_sentences_merged=data['train_src_sentences_merged'],
                        val_src_sentences_merged=data['val_src_sentences_merged'],
                        test_src_sentences_merged=data['test_src_sentences_merged'],
                        y_val=data['y_val'],
                        y_test=data['y_test'])

    end = time.time()

    print "time spent: ", (end - begin), "s"
