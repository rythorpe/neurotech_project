# -*- coding: utf-8 -*-
"""Lempel-Ziv complexity for a binary sequence, in naive Python code.

- How to use it? From Python, it's easy:

>>> from research.lempel_ziv_complexity import lempel_ziv_complexity
>>> s = '1001111011000010'
>>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
8

- Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.

- MIT Licensed, (C) 2017-2019 Lilian Besson (Naereen)
  https://GitHub.com/Naereen/Lempel-Ziv_Complexity
"""
from __future__ import print_function

__author__ = "Lilian Besson (Naereen)"
__version__ = "0.2.2"


from collections import OrderedDict


def lempel_ziv_decomposition(sequence):
    r""" Manual implementation of the Lempel-Ziv decomposition.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_decomposition(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    ['1', '0', '01', '11', '10', '110', '00', '010']

    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 8`: :math:`s = 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010`.

    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.


    Other examples:

    >>> lempel_ziv_decomposition('1010101010101010')
    ['1', '0', '10', '101', '01', '010', '1010']
    >>> lempel_ziv_decomposition('1001111011000010000010')
    ['1', '0', '01', '11', '10', '110', '00', '010', '000']
    >>> lempel_ziv_decomposition('100111101100001000001010')
    ['1', '0', '01', '11', '10', '110', '00', '010', '000', '0101']

    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """
    sub_strings = OrderedDict()
    n = len(sequence)

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        # print(sub_str, ind, inc)
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings[sub_str] = 0
            ind += inc
            inc = 1
            # print("Adding", sub_str)
    return list(sub_strings)


def lempel_ziv_complexity(sequence, sub_strings=None):
    r""" Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    8

    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 8`: :math:`s = 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010`.

    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.


    Other examples:

    >>> lempel_ziv_complexity('1010101010101010')  # 1, 0, 10, 101, 01, 010, 1010
    7
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000
    9
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000, 0101
    10

    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """
    if sub_strings is None:
        sub_strings = set()

    if isinstance(sequence, list):
        if len(sequence) > 1:
            component_seq = sequence[:-1]
            _, sub_strings = lempel_ziv_complexity(component_seq,
                                                   sub_strings=sub_strings)

        sequence = sequence[-1]

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings), sub_strings


def lempel_ziv_complexity_md(seq_components, sub_strings=None):
    r""" Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    8

    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 8`: :math:`s = 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010`.

    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.


    Other examples:

    >>> lempel_ziv_complexity('1010101010101010')  # 1, 0, 10, 101, 01, 010, 1010
    7
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000
    9
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000, 0101
    10

    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """
    if sub_strings is None:
        sub_strings = set()

    if isinstance(seq_components, list):
        sequence = seq_components[0]
    else:
        sequence = seq_components

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1

    if len(seq_components) > 1 and isinstance(seq_components, list):
        _, sub_strings = lempel_ziv_complexity(seq_components[1:],
                                               sub_strings=sub_strings)
    return len(sub_strings), sub_strings



# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    #from doctest import testmod
    #print("\nTesting automatically all the docstring written in each functions of this module :")
    #testmod(verbose=True)

    seq_components = ['10110100111001010010100000011',
                      '00100111000110001011010000110',
                      '11010011010000101100001011010']
    complexity, _ = lempel_ziv_complexity_md(seq_components)
    print(f'Lempel-Ziv complexity for {seq_components}: {complexity}')

    agg_seq = seq_components[0] + seq_components[1] + seq_components[2]
    complexity, _ = lempel_ziv_complexity_md(agg_seq)
    print(f'Lempel-Ziv complexity for {agg_seq}: {complexity}')
