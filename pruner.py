import argparse
from collections import Counter
from string import punctuation
from typing import List, Set

import numpy as np
import matplotlib.pyplot as plt


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='Pruner', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help='Input text.')
    parser.add_argument('--output', type=str, help='Output of pruned text.')
    parser.add_argument('--statistics', action='store_true',
                        help="Print out statistics and additionally"
                             "create file with word frequencies called 'statistics'")

    subparsers = parser.add_subparsers(help='Various methods for pruning word from the input text.')

    prune_mc = subparsers.add_parser('mc', help='Removes most common words.')
    prune_mc.add_argument('--n', type=int, default=-1,
                          help='Removes the n most common words from the text.')

    prune_freq = subparsers.add_parser('word_freq', help='Remove words by their frequencies.')
    prune_freq.add_argument('--at-most', type=int, default=-1,
                            help='Remove words that occur more than n times in the input corpus.'
                                 'Set to 0 to turn this off.')
    return parser


def print_word_statistics(text: List[str]) -> None:
    word_counter = Counter()
    line_lengths = []

    for line in text:
        word_counter.update(line.split())
        line_lengths.append(len(line))

    frequencies = list(word_counter.values())
    avg_word_freq = np.mean(frequencies)
    max_word_freq = np.max(frequencies)

    print("Average word count/sentence: {}".format(np.mean(line_lengths)))
    print("Maximum word count/sentence: {}".format(np.max(line_lengths)))
    print("Average word frequency: {}".format(avg_word_freq))
    print("Maximum word frequency: {}".format(max_word_freq))

    # compute plot
    # freq_of_freq = Counter(frequencies)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot('111')
    #
    # X = np.arange(max_word_freq+1)
    # Y = np.zeros(max_word_freq+1)
    #
    # for x, freq in freq_of_freq.items():
    #     Y[x] = freq
    #
    # print(X)
    # print(Y)
    # ax.bar(X, Y)
    #
    # fig.savefig('/home/duc/example.png')


def prune_text(text: List[str], filter_set: Set[str]) -> List[str]:
    pruned = []
    for line in text:
        pruned_line = [token for token in line.split() if token not in filter_set]
        pruned.append(' '.join(pruned_line))
    return pruned


def remove_punctuations(text: List[str]) -> List[str]:
    return prune_text(text, punctuation)


def remove_words_by_frequencies(text: List[str], min_frequency: int) -> List[str]:
    word_counter = Counter()
    for line in text:
        word_counter.update(line.split())

    high_freq_words = set()
    for word, freq in word_counter.items():
        if freq >= min_frequency:
            high_freq_words.add(word)

    return prune_text(text, high_freq_words)


if __name__ == '__main__':
    args = create_parser().parse_args()

    with open('test_data', encoding='UTF-8') as input_data:
        text = input_data.read().splitlines()

    # TODO: create a decorator for printing headlines

    print('Original input text statistics:\n-------------------------')
    print_word_statistics(text)

    text = remove_punctuations(text)

    print('\nAfter punctuations removal:\n-------------------------')
    print_word_statistics(text)

    text = remove_words_by_frequencies(text, 10)

    print('\nAfter punctuations removal:\n-------------------------')
    print_word_statistics(text)
