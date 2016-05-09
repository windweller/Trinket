import tensorflow as tf
from util import load_vocab

flags = tf.flags

# flags.DEFINE_string()



if __name__ == '__main__':
    word_idx_map, idx_word_map = load_vocab('')