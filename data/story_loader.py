"""
A Loader class
that get initialized by batch_size
and can be iterated by batches
"""

import numpy as np


class StoryLoader(object):
    def __init__(self, npz_file, batch_size, src_seq_len, tgt_seq_len,
                 train_frac=0.95, valid_frac=0.05, mode='merged', debug=False):
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

            _, self.splits['pre_train_tgt'] = self._split_batch(
                train_sentence5,
                tgt_seq_len)

            if debug:
                from model.tf.util import decode_sentences, load_vocab
                word_idx_map, idx_word_map = load_vocab(
                    '/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/allen/trident/data/story_vocab.json')

                print decode_sentences(self.splits['pre_train'][0, 1, :], idx_word_map)
                print decode_sentences(self.splits['train_tgt'][0, 1, :], idx_word_map)

            # split val into train and val again
            _, self.splits['train_tgt1'] = self._split_batch(
                val_sentence5[:train_par * batch_size],
                tgt_seq_len)
            _, self.splits['train_tgt2'] = self._split_batch(
                val_sentence6[:train_par * batch_size],
                tgt_seq_len)

            _, self.splits['val_tgt1'] = self._split_batch(
                val_sentence5[train_par * batch_size:],
                tgt_seq_len)
            _, self.splits['val_tgt2'] = self._split_batch(
                val_sentence6[train_par * batch_size:],
                tgt_seq_len)

            _, self.splits['test_tgt1'] = self._split_batch(
                test_sentence5,
                tgt_seq_len)
            _, self.splits['test_tgt2'] = self._split_batch(
                test_sentence6,
                tgt_seq_len)

            # there's an ugly fix to preprocessing error on y_val
            _, self.splits['y_train'] = self._split_batch(
                y_val[0][:train_par * batch_size], 0)

            _, self.splits['y_val'] = self._split_batch(
                y_val[0][train_par * batch_size:], 0)

            _, self.splits['y_test'] = self._split_batch(
                y_test, 0)

        elif self.mode == 'seperate':
            raise NotImplementedError

    def _split_batch(self, sens, max_seq_len):
        """

        Parameters
        ----------
        sens: np.array(corpus_length, num_words_per_sen)
            This should be the raw sentence read from npz

        max_seq_len: int
            if set to 0, then we are processing y labels
        Returns
        -------
        num_batches, np.array(num_batches, batch_size, src_seq_len)

        """

        if max_seq_len == 0:
            dim = 1
        else:
            dim = sens.shape[1]

        remainder = sens.shape[0] % self.batch_size
        partitions = sens.shape[0] / self.batch_size
        if remainder != 0:
            sens = sens[:-remainder]

        if max_seq_len != 0:
            s = sens.reshape((partitions, self.batch_size, dim))[:, :, :max_seq_len]
        else:
            s = sens.reshape((partitions, self.batch_size, dim))

        return partitions, s

    def get_pretrain_batch(self, batch_id):
        """

        Parameters
        ----------
        batch_id

        Returns
        -------
        tuple: (src_sen_merged, tgt_sen)
            src_sen_merged: (batch_size x max_src_len)
            tgt_sen: (batch_size x max_tgt_len)
        """
        if self.mode == 'merged':
            assert batch_id <= self.pre_train_num_batches
            x = self.splits['pre_train'][batch_id]
            y = self.splits['pre_train_tgt'][batch_id]
            return x, y
        else:
            raise NotImplementedError

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
            to give back either (merged_src, y, label) or ([src1, src2...], y, label)

        """
        x = self.splits[split + '_src'][batch_id]
        # TODO: why am I feeding in both sentences?
        # TODO: binary prediction, we only need 1 sentence
        y = self.splits[split + '_tgt1'][batch_id]
        label = self.splits['y_' + split][batch_id]

        return x, y, label

if __name__ == '__main__':
    # offers some testing for the class
    loader = StoryLoader('/Users/Aimingnie/Documents/School' +
                         '/Stanford/CS 224N/DeepLearning/allen/trident/data/story_processed.npz',
                         batch_size=50, src_seq_len=65,
                         tgt_seq_len=20, mode='merged')

    loader.get_pretrain_batch(1)
    loader.get_batch('train', 1)
