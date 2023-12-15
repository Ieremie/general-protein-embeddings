"""
Adapted from https://github.com/tbepler/prose
"""

from __future__ import print_function, division
import random
import numpy as np
from itertools import product

class Alphabet:
    def __init__(self, encoding, missing=20):
        # mapping the char -> ASCII value
        self.chars = np.frombuffer(b'ARNDCQEGHILKMFPSTWYVXOUBZ', dtype=np.uint8)

        # we assume that any wrong AA label is mapped to missing type 20
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        self.encoding[self.chars] = encoding
        self.size = encoding.max() + 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()


class Uniprot21(Alphabet):
    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))

        # O as K
        # U as C
        # B and Z as X (unknown)
        encoding[21:] = [11, 4, 20, 20]  # O, U, B, Z
        self.groupings = {k: v for k, v in zip(chars, encoding)}
        super(Uniprot21, self).__init__(encoding=encoding, missing=20)


# get the alphabet based on string name
def get_alphabet(name):
    if name == 'uniprot21':
        return Uniprot21()
    else:
        raise ValueError('Unknown alphabet: {}'.format(name))
