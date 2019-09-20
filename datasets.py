# -*- coding: utf-8 -*-

import math
import io

from torchtext import data
from torchtext.data import Dataset
from torchtext.data.batch import Batch


class LanguageModelingDataset(data.Dataset):
    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', **kwargs):
        fields = [('src', text_field)]
        text = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                text += text_field.preprocess(line)
                if newline_eos:
                    text.append(u'<eos>')

        examples = [data.Example.fromlist([text], fields)]
        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)


class BPTTIterator(data.Iterator):
    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil((len(self.dataset[0].src) / self.batch_size - 1)
                         / self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].src
        TEXT = self.dataset.fields['src']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device)
        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('src', TEXT), ('tgt', TEXT)])

        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + 1 + seq_len]
                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    src=batch_text,
                    tgt=batch_target)
            if not self.repeat:
                return

