import numpy as np
import tensorflow as tf

def get_name_list(data, min_length=0, max_length=None, stripped=False):
    results = []

    if max_length is None:
        max_length = max([len(d.strip()) for d in data])
    data.seek(0)

    for n in data:
        line = n.strip() if stripped else n
        if min_length <= len(line) <= max_length:
            results.append(line)

    return results, max_length


def get_the_longest(data):
    mlen = max([len(d) for d in data])
    the_longests = [d for d in data if mlen == len(d)]
    return the_longests


def get_vocab(names):
    vocab = sorted(set(''.join([n for n in names])))
    return vocab
