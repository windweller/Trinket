import os
import string
from os.path import join as pjoin

'''
path settings
'''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

STORY_DATA_PATH = pjoin(ROOT_DIR, 'data/story_processed.npz')
VOCAB_PATH = pjoin(ROOT_DIR, 'data/story_vocab.json')
EMBED_PATH = pjoin(ROOT_DIR, 'data')