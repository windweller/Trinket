"""
A Loader class
that get initialized by batch_size
and can be iterated by batches
"""

import numpy as np
import sys


class StoryLoader(object):
    def __init__(self, npz_file, batch_size, src_seq_len, tgt_seq_len, pretrain_split=[0.9, 0.05, 0.05],
                 train_frac=0.9, valid_frac=0.05, mode='merged', debug=False):
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
        self.pretrain_split = pretrain_split
        self.pretrain_frac = pretrain_split[0]
        self.preval_frac = pretrain_split[1]
        self.pretest_frac = pretrain_split[2]
        self.train_frac = train_frac
        self.valid_frac = valid_frac
        self.mode = mode

        if self.mode == 'merged':
            train_src_sens_merged = self.data_pt['train_src_sentences_merged']
            val_src_sens_merged = self.data_pt['val_src_sentences_merged']
            test_src_sens_merged = self.data_pt['test_src_sentences_merged']

            self.splits = dict()

            # ===== splitting pretrain =======
            N = train_src_sens_merged.shape[0]
            partitions = N / self.batch_size
            train_par = int(np.round(partitions * self.pretrain_frac))
            val_par = int(np.round(partitions * self.preval_frac))
            self.pretrain_num_batches, self.splits['pre_train'] = self._split_batch(
                train_src_sens_merged[:train_par * batch_size],
                src_seq_len)
            self.preval_num_batches, self.splits['pre_val'] = self._split_batch(
                train_src_sens_merged[train_par * batch_size: (train_par + val_par) * batch_size],
                src_seq_len)
            self.pretest_num_batches, self.splits['pre_test'] = self._split_batch(
                train_src_sens_merged[(train_par + val_par) * batch_size:],
                src_seq_len)

            train_sentence5 = self.data_pt['train_sentence5']

            _, self.splits['pre_train_tgt'] = self._split_batch(
                train_sentence5[:train_par * batch_size],
                tgt_seq_len)

            _, self.splits['pre_val_tgt'] = self._split_batch(
                train_sentence5[train_par * batch_size: (train_par + val_par) * batch_size],
                tgt_seq_len)

            _, self.splits['pre_test_tgt'] = self._split_batch(
                train_sentence5[(train_par + val_par) * batch_size:],
                tgt_seq_len)

            # ======= splitting valid/test =====
            # make sure we don't lose a batch due to rounding
            N = val_src_sens_merged.shape[0] + test_src_sens_merged.shape[0]
            partitions = N / self.batch_size
            train_par = int(np.round(partitions * train_frac))
            val_par = int(np.round(partitions * valid_frac))

            assert train_frac + valid_frac < 1.

            whole = np.concatenate((val_src_sens_merged, test_src_sens_merged), axis=0)

            self.train_num_batches, self.splits['train_src'] = self._split_batch(
                whole[:train_par * batch_size],
                src_seq_len)

            self.val_num_batches, self.splits['val_src'] = self._split_batch(
                whole[train_par * batch_size: (train_par + val_par) * batch_size],
                src_seq_len)

            self.test_num_batches, self.splits['test_src'] = self._split_batch(
                whole[(train_par + val_par) * batch_size:],
                src_seq_len
            )

            val_sentence5 = self.data_pt['val_sentence5']
            val_sentence6 = self.data_pt['val_sentence6']
            test_sentence5 = self.data_pt['test_sentence5']
            test_sentence6 = self.data_pt['test_sentence6']

            y_val = self.data_pt['y_val'],
            y_test = self.data_pt['y_test']

            if debug:
                from model.tf.util import decode_sentences, load_vocab
                word_idx_map, idx_word_map = load_vocab(
                    '/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/allen/trident/data/story_vocab.json')

                print decode_sentences(self.splits['pre_train'][0, 1, :], idx_word_map)
                print decode_sentences(self.splits['train_tgt'][0, 1, :], idx_word_map)

            # split val into train and val again
            whole_sentence5 = np.concatenate((val_sentence5, test_sentence5), axis=0)
            whole_sentence6 = np.concatenate((val_sentence6, test_sentence6), axis=0)

            # pad up sentence6 (because it's 18 instead of 20)
            whole_sentence6 = np.pad(whole_sentence6, ((0,0), (0,2)), mode='constant', constant_values=0)

            self.train_num_batches, self.splits['train_tgt1'] = self._split_batch(
                whole_sentence5[:train_par * batch_size],
                tgt_seq_len)
            _, self.splits['train_tgt2'] = self._split_batch(
                whole_sentence6[:train_par * batch_size], tgt_seq_len)

            self.val_num_batches, self.splits['val_tgt1'] = self._split_batch(
                whole_sentence5[train_par * batch_size: (train_par + val_par) * batch_size],
                tgt_seq_len)
            _, self.splits['val_tgt2'] = self._split_batch(
                whole_sentence6[train_par * batch_size: (train_par + val_par) * batch_size],
                tgt_seq_len)

            self.test_num_batches, self.splits['test_tgt1'] = self._split_batch(
                whole_sentence5[(train_par + val_par) * batch_size:],
                tgt_seq_len)
            _, self.splits['test_tgt2'] = self._split_batch(
                whole_sentence6[(train_par + val_par) * batch_size:],
                tgt_seq_len)

            # there's an ugly fix to preprocessing error on y_val
            all_ys = np.concatenate((y_val[0], y_test), axis=0)

            _, self.splits['y_train'] = self._split_batch(
                all_ys[:train_par * batch_size], 0)

            _, self.splits['y_val'] = self._split_batch(
                all_ys[train_par * batch_size: (train_par + val_par) * batch_size], 0)

            _, self.splits['y_test'] = self._split_batch(
                all_ys[(train_par + val_par) * batch_size:], 0)

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

    def get_pretrain_batch(self, split, batch_id):
        """

        Parameters
        ----------
        batch_id
        split: {'train', 'val', 'test'}

        Returns
        -------
        tuple: (src_sen_merged, tgt_sen)
            src_sen_merged: (batch_size x max_src_len)
            tgt_sen: (batch_size x max_tgt_len)
        """
        if self.mode == 'merged':
            # pre_train_tgt
            x = self.splits['pre_' + split][batch_id]
            y = self.splits['pre_' + split + '_tgt'][batch_id]
            return x, y
        else:
            raise NotImplementedError

    def get_batch(self, split, batch_id, label_size=2):
        """

        Parameters
        ----------
        split: {'train', 'val', 'test'}
        batch_id: int

        Returns
        -------
        tuple
            we will decide based on the "self.mode" information
            to give back either (merged_src, (y1, y2), label) or ([src1, src2...], y, label)

        """
        x = self.splits[split + '_src'][batch_id]

        y = self.splits[split + '_tgt1'][batch_id]
        y_2 = self.splits[split + '_tgt2'][batch_id]
        label = self.splits['y_' + split][batch_id]

        # bring down the label range from 1, 2 to 0, 1
        return x, (y, y_2), label.flatten() - 1

    def get_w2v_embed(self):
        """
        Returns
        -------
        |V| x D shape matrix
        The default word embedding matrix
        """
        return self.data_pt['W_embed']


if __name__ == '__main__':
    # offers some testing for the class
    loader = StoryLoader('/Users/Aimingnie/Documents/School' +
                         '/Stanford/CS 224N/DeepLearning/allen/trident/data/story_processed.npz',
                         batch_size=50, src_seq_len=65,
                         tgt_seq_len=20, mode='merged')

    loader.get_pretrain_batch('train', 1)
    loader.get_pretrain_batch('val', 1)

    print "pretrain num: ", loader.pretrain_num_batches
    print "preval num: ", loader.preval_num_batches
    print "pretest num: ", loader.pretest_num_batches

    loader.get_batch('train', 1)
    x, (y, y_2), label = loader.get_batch('val', 2)

    from model.tf.util import decode_sentences, load_vocab

    word_idx_map, idx_word_map = load_vocab(
        '/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/allen/trident/data/story_vocab.json')

    # remember the range of y_label is 1 and 2
    print decode_sentences(x[1], idx_word_map)
    print decode_sentences(y[1], idx_word_map)
    print decode_sentences(y_2[1], idx_word_map)
    print label[1]