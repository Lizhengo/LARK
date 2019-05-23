#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import types
import gzip
import logging
import re
import six
import collections
import tokenization

import paddle
import paddle.fluid as fluid

from batching import prepare_batch_data


class ErnieDataReader(object):
    def __init__(self,
                 filelist,
                 vocab_path,
                 batch_size=4096,
                 max_seq_len=512,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 is_test=False,
                 in_tokens=False,
                 generate_neg_sample=False,
                 is_bidirection=False):

        self.vocab = self.load_vocab(vocab_path)
        self.filelist = filelist
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.is_test = is_test
        self.generate_neg_sample = generate_neg_sample
        self.in_tokens = in_tokens
        self.is_bidirection = is_bidirection
        if self.in_tokens:
            assert self.batch_size > 100, "Current batch size means total token's number, \
                                       it should not be set to too small number."

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file, self.mask_type

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().split(";")
        assert len(line) == 5, "One sample must have 5 fields!"
        (token_ids, sent_ids, pos_ids, seg_labels, label) = line
        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        seg_labels = [int(seg_label) for seg_label in seg_labels.split(" ")]
        assert len(token_ids) == len(sent_ids) == len(pos_ids) == len(
            seg_labels
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels)"
        label = int(label)
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, label, seg_labels]

    def read_file(self, file):
        assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        f = gzip.open(file, "rb").readlines()
        if self.shuffle_files:
            print("begin shuffle data")
            np.random.shuffle(f)
            print("end shuffle data")
        for line in f:
            parsed_line = self.parse_line(
                line, max_seq_len=self.max_seq_len)
            if parsed_line is None:
                continue
            yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negtive samples using pos_samples

            Args:
                pos_samples: list of positive samples
            
            Returns:
                neg_samples: list of negtive samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        def split_sent(sample, max_len, sep_id):
            token_seq, type_seq, pos_seq, label, seg_labels = sample
            sep_index = token_seq.index(sep_id)
            left_len = sep_index - 1
            if left_len <= max_len:
                return (token_seq[1:sep_index], seg_labels[1:sep_index])
            else:
                return [
                    token_seq[sep_index + 1:-1], seg_labels[sep_index + 1:-1]
                ]

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            left_tokens, left_seg_labels = split_sent(
                pos_samples[i], (self.max_seq_len - 3) // 2, self.sep_id)
            right_tokens, right_seg_labels = split_sent(
                pos_samples[pair_index],
                self.max_seq_len - 3 - len(left_tokens), self.sep_id)

            token_seq = [self.cls_id] + left_tokens + [self.sep_id] + \
                    right_tokens + [self.sep_id]
            if len(token_seq) > self.max_seq_len:
                miss_num += 1
                continue
            type_seq = [0] * (len(left_tokens) + 2) + [1] * (len(right_tokens) +
                                                             1)
            pos_seq = range(len(token_seq))
            seg_label_seq = [-1] + left_seg_labels + [-1] + right_seg_labels + [
                -1
            ]

            assert len(token_seq) == len(type_seq) == len(pos_seq) == len(seg_label_seq), \
                    "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append([token_seq, type_seq, pos_seq, 0, seg_label_seq])

        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negtive samples and positive samples
            
            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

            Returns:
                sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))

    def data_generator(self):
        """
        data_generator
        """
        files = open(self.filelist).readlines()
        self.total_file = len(files)
        assert self.total_file > 0, "[Error] data_dir is empty"

        def wrapper():
            def reader():
                for epoch in range(self.epoch):
                    self.current_epoch = epoch + 1
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        file, mask_word_prob = file.strip().split("\t")
                        mask_word = (np.random.random() < float(mask_word_prob))
                        self.current_file_index = index + 1
                        self.current_file = file
                        if mask_word:
                            self.mask_type = "mask_word"
                        else:
                            self.mask_type = "mask_char"

                        sample_generator = self.read_file(file)
                        if not self.is_test and self.generate_neg_sample:
                            sample_generator = self.mixin_negtive_samples(
                                sample_generator)
                        for sample in sample_generator:
                            if sample is None:
                                continue
                            sample.append(mask_word)
                            yield sample

            def batch_reader(reader, batch_size):
                batch, total_token_num, max_len = [], 0, 0
                for parsed_line in reader():
                    token_ids, sent_ids, pos_ids, label, seg_labels, mask_word = parsed_line
                    max_len = max(max_len, len(token_ids))
                    if self.in_tokens:
                        to_append = (len(batch) + 1) * max_len <= batch_size
                    else:
                        to_append = len(batch) < batch_size

                    if to_append:
                        batch.append(parsed_line)
                        total_token_num += len(token_ids)
                    else:
                        yield batch, total_token_num
                        batch, total_token_num, max_len = [parsed_line], len(
                            token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(reader,
                                                            self.batch_size):
                yield prepare_batch_data(
                    batch_data,
                    total_token_num,
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False,
                    is_bidirection=self.is_bidirection)

        return wrapper


if __name__ == "__main__":
    pass
