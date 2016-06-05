import numpy as np
import csv
import nltk
from collections import Counter
import re
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

###################### read data ######################

# validation + test concat

split = {
	'training': 3350,
	'validation': 200,
	'test': 150
}
train_start = 0
train_end = split['training']
val_start = train_end
val_end = train_end + split['validation']
test_start = val_end
test_end = val_end + split['test']

orig_val_file = 'data/story_corpus_16/cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
orig_test_file = 'data/story_corpus_16/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'

orig_cloze_data = []
for row in csv.DictReader(open(orig_val_file, 'Ub')):
	orig_cloze_data.append(row)
for row in csv.DictReader(open(orig_test_file, 'Ub')):
	orig_cloze_data.append(row)

train = orig_cloze_data[train_start:train_end]
val = orig_cloze_data[val_start:val_end]
test = orig_cloze_data[test_start:test_end]

# def ngrams(tree):
#     return Counter()

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

# def word_cross_product_phi(t1, t2):
#     return Counter([(w1, w2) for w1, w2 in itertools.product(t1, t2)])

def embed(set_data):

	y = []
	X_as_dicts = []
	dimensions = set()

	for example in set_data:
		# print 'Example ID:', example['InputStoryid']

		context_keys = ['InputSentence' + str(i) for i in range(1,5)]
		context_sentences = [example[key] for key in context_keys]
		context = ' '.join(context_sentences)
		ending_keys = ['RandomFifthSentenceQuiz' + str(i) for i in range(1,3)]
		ending_sentences = [example[key] for key in ending_keys]
		first_ending = ending_sentences[0]

		context_tokens = clean_str(context).split(' ')
		ending_tokens = clean_str(first_ending).split(' ')

		def collect_ngrams(tokens):
			unigrams = Counter([(t,) for t in tokens])
			bigrams = Counter(nltk.ngrams(tokens,2))
			trigrams = Counter(nltk.ngrams(tokens,3))
			features = unigrams + bigrams + trigrams
			return features

		context_ngrams = [('CONTEXT',)+ngram for ngram in collect_ngrams(context_tokens)]
		ending_ngrams = [('ENDING',)+ngram for ngram in collect_ngrams(ending_tokens)]

		dimensions.update(context_ngrams)
		dimensions.update(ending_ngrams)

		## add sentiment feature
		## event chain

		features = Counter(context_ngrams + ending_ngrams)
		X_as_dicts.append(features)
		y.append(int(example['AnswerRightEnding'])-1)
	return (np.array(y), X_as_dicts, dimensions)

train_y, train_X_as_dicts, train_dims = embed(train)
print train_y[:10]
val_y, val_X_as_dicts, val_dims = embed(val)
test_y, test_X_as_dicts, test_dims = embed(test)

dimensions = train_dims.union(val_dims.union(test_dims))

v = DictVectorizer(sparse=True)
X_all = v.fit_transform(train_X_as_dicts + val_X_as_dicts + test_X_as_dicts)
X_train = X_all[train_start:train_end]
X_val = X_all[val_start:val_end] 

# mod = LogisticRegression(fit_intercept=True)
# mod = svm.SVC(kernel='rbf', verbose=True)
mod = svm.SVC(kernel='linear', verbose=True)
mod.fit(X_train, train_y)
predictions = mod.predict(X_train)
print(classification_report(train_y, predictions))
predictions = mod.predict(X_val)
print(classification_report(val_y, predictions))