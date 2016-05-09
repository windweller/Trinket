"""
A Loader class
that get initialized by batch_size
and can be iterated by batches
"""

import numpy as np
from model.tf.util import decode_sentences, load_vocab

class StoryLoader(object):
    def __init__(self, npz_file, batch_size, src_seq_len, tgt_seq_len,
                 train_frac=0.95, valid_frac=0.05, mode='merged'):
        """
        Parameters
        ----------
        npz_file : str
            the address of npz_file
        batch_size: int
        seq_len: int
            We truncate anything out of this range
        mode: {'merged', 'seperate'}
            merged - load all story plot (4 sentences) as one encoding
            seperate - load story plot seperately (NotImplemented)
        """
        self.data_pt = np.load(npz_file)
        self.batch_size = batch_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.train_frac = train_frac
        self.valid_frac = valid_frac
        self.mode = mode

        if self.mode == 'merged':
            train_src_sens_merged = self.data_pt['train_src_sentences_merged']
            val_src_sens_merged = self.data_pt['val_src_sentences_merged']
            test_src_sens_merged = self.data_pt['test_src_sentences_merged']

            self.splits = dict()
            self.pre_train_num_batches, self.splits['pre_train'] = self._split_batch(train_src_sens_merged,
                                                                                     src_seq_len)
            # make sure we don't lose a batch due to rounding
            N = val_src_sens_merged.shape[0]
            partitions = N / self.batch_size
            train_par = int(np.round(partitions * 0.95))

            self.train_num_batches, self.splits['train_src'] = self._split_batch(
                val_src_sens_merged[:train_par * batch_size],
                src_seq_len)
            self.val_num_batches, self.splits['val_src'] = self._split_batch(
                val_src_sens_merged[train_par * batch_size:],
                src_seq_len)

            self.test_num_batches, self.splits['test_src'] = self._split_batch(
                test_src_sens_merged,
                src_seq_len
            )

            train_sentence5 = self.data_pt['train_sentence5']
            val_sentence5 = self.data_pt['val_sentence5']
            val_sentence6 = self.data_pt['val_sentence6']
            test_sentence5 = self.data_pt['test_sentence5']
            test_sentence6 = self.data_pt['test_sentence6']

            y_val = self.data_pt['y_val'],
            y_test = self.data_pt['y_test']

            _, self.splits['train_tgt'] = self._split_batch(
                train_sentence5,
                tgt_seq_len)

            word_idx_map, idx_word_map = load_vocab('/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/allen/trident/data/story_vocab.json')

            print decode_sentences(self.splits['pre_train'][0,1,:], idx_word_map)
            print decode_sentences(self.splits['train_tgt'][0,1,:], idx_word_map)

        elif self.mode == 'seperate':
            raise NotImplementedError

    def _split_batch(self, sens, max_seq_len):
        """

        Parameters
        ----------
        sens: np.array(corpus_length, num_words_per_sen)
            This should be the raw sentence read from npz
        Returns
        -------
        num_batches, np.array(num_batches, batch_size, src_seq_len)

        """

        dim = sens.shape[1]

        remainder = sens.shape[0] % self.batch_size
        partitions = sens.shape[0] / self.batch_size
        if remainder != 0:
            sens = sens[:-remainder]

        s = sens.reshape((partitions, self.batch_size, dim))[:, :, :max_seq_len]

        return partitions, s

    def get_pretrain_batch(self, batch_id):
        pass

    def get_batch(self, split, batch_id):
        """

        Parameters
        ----------
        split: {'train', 'val', 'test'}
        batch_id: int

        Returns
        -------
        tuple
            we will decide based on the "self.mode" information
            to give back either (merged_src, y) or

        """
        pass


if __name__ == '__main__':
    # offers some testing for the class
    loader = StoryLoader('/Users/Aimingnie/Documents/School' +
                         '/Stanford/CS 224N/DeepLearning/allen/trident/data/snli_processed.npz',
                         batch_size=50, src_seq_len=65,
                         tgt_seq_len=20)
