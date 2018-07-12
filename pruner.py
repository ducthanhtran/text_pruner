import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='Pruner', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help='Input text.')
    parser.add_argument('--output', type=str, help='Output of pruned text.')
    parser.add_argument('--statistics', action=store_true,
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


def compute_statistics(text: str):
    word_counter = Counter(text.split())

    frequencies = list(word_counter.values())
    avg_word_freq = np.mean(frequencies)
    max_word_freq = np.max(frequencies)

    print("Average word frequency: {}".format(avg_word_freq))
    print("Maximum word frequency: {}".format(max_word_freq))

    # compute plot
    freq_of_freq = Counter(frequencies)

    fig = plt.figure()
    ax = fig.add_subplot('111')

    X = np.arange(max_word_freq+1)
    Y = np.zeros(max_word_freq+1)

    for x, freq in freq_of_freq.items():
        Y[x] = freq

    print(X)
    print(Y)
    ax.hist(X, Y)

    fig.savefig('/u/tran/example.png')




if __name__ == '__main__':
    with open ('test_data', encoding='UTF-8') as input_data:
        text = input_data.read()

    compute_statistics(text)