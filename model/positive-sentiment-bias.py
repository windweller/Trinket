#!/usr/bin/env python

# super dumb model that just...
#   * combines the whole story into one sentence (for each conclusion),
#   * runs through corenlp,
#   * gets a sentiment tag,
#   * picks the one that's more strongly negative or positive

# validation: 0.549732620321

import codecs, csv
import os
import sys
import json
import random
import sys
import re

val_file = '../data/story_corpus_16/cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
test_file = '../data/story_corpus_16/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
val_caching_file = '../data/corenlp-parses/validation.txt'
test_caching_file = '../data/corenlp-parses/test.txt'

if len(sys.argv)>1 and sys.argv[1]=='test':
    print 'running on test data'
    input_file = test_file
    caching_file = test_caching_file
else:
    print 'running on validation data'
    input_file = val_file
    caching_file = val_caching_file


cached = []
with codecs.open(caching_file, 'r', 'utf-8') as f:
    for line in f:
        cached.append(line[:-1])
    f.close()

caching_writer = codecs.open(caching_file, 'a', 'utf-8')


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def removequote(s):
    return re.sub('"', '', s);

# combine the whole story into one sentence for each conclusion

correct_stories = []
incorrect_stories = []
with codecs.open(input_file, 'rU', 'utf-8') as csvfile:
    reader = unicode_csv_reader(csvfile)
    i = 0
    for row in reader:
        if i>1:
            storyid, sent0, sent1, sent2, sent3, conc1, conc2, correct = row
            if correct == '1':
                correct_conclusion = conc1
                incorrect_conclusion = conc2
            else:
                correct_conclusion = conc2
                incorrect_conclusion = conc1
            story_stem = ', '.join([sent[:-1] for sent in [sent0, sent1, sent2, sent3]]) + ', '
            correct_story = story_stem + correct_conclusion
            incorrect_story = story_stem + incorrect_conclusion
            correct_stories.append(correct_story)
            incorrect_stories.append(incorrect_story)
        i += 1

assert(len(correct_stories)==len(incorrect_stories))

sentence_pairs = zip(correct_stories, incorrect_stories)

w = codecs.open('../data/story_as_sentence_for_dumb_sentiment_model/validation-caching.csv', 'wb', 'utf-8')
w.write('')
w.close()
w = codecs.open('../data/story_as_sentence_for_dumb_sentiment_model/validation-caching.csv', 'ab', 'utf-8')

def get_sentiment(sentence, i):
    if i < len(cached):
        filestring = cached[i]
    else:
        print "sentence", sentence
        ## java -cp "*:lib/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,parse,sentiment -file input.txt
        ## java -mx4g -cp "*:lib/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
        ## ## java -mx4g -cp "javanlp-core.jar:stanford-english-corenlmodels-current.jar:lib/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
        print('wget --post-data "' + removequote(sentence.encode('ascii', errors='ignore')) + '" \'localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,parse,sentiment", "outputFormat": "json"}\' -O - > ../data/story_as_sentence_for_dumb_sentiment_model/tmp.json')
        os.system('wget --post-data "' + removequote(sentence.encode('ascii', errors='ignore')) + '" \'localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,parse,sentiment", "outputFormat": "json"}\' -O - > ../data/story_as_sentence_for_dumb_sentiment_model/tmp.json')

        filestring = open('../data/story_as_sentence_for_dumb_sentiment_model/tmp.json', 'rb').read()

    if len(filestring) > 10:
        caching_writer.write(filestring + '\n')
    data = json.loads(filestring)
    sentiment = data['sentences'][0]['sentimentValue']
    return sentiment

results = []
for i in range(len(sentence_pairs)):
    print i
    correct, incorrect = sentence_pairs[i]
    correct_sentiment = get_sentiment(correct, i*2)
    incorrect_sentiment = get_sentiment(incorrect, i*2+1)
  ## oops, this picks the most positive story always...
    if correct_sentiment > incorrect_sentiment:
        choice = 1
    elif incorrect_sentiment > correct_sentiment:
        choice = 0
    else:
        choice = 0.5
        # choice = random.randint(0, 1)
    row = [str(x) for x in [i, correct_sentiment, incorrect_sentiment, choice]]
    w.write(",".join(row) + '\n')
    results.append(choice)

    print float(sum(results))/len(results)