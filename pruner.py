import argparse
from collections import Counter
from itertools import chain
from string import punctuation
from typing import Callable, List, Set, Tuple

import numpy as np


DELETED_SENTENCE_TOKEN = '<DELETED_SENTENCE>'


# TODO: introduce argument groups for all strategies.
# TODO: Make --help nicer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input text.')
    parser.add_argument('--output', type=str, required=True, help='Output of pruned text.')
    parser.add_argument('--strategy', choices=['mc', 'freq', 'sent-length', 'none'],
                        help='Various methods for pruning word from the input text.\n'
                             'none: only punctuations are removed\n'
                             'mc: most-common words are pruned\n'
                             'freq: words that occur more than a certain frequency\n'
                             'sent-length: prune most-common words in long sentences such that a certain'
                             'sentence length is achieved ('
                             'can fail due to low sentence length parameter)')

    parser.add_argument('--mc-number', type=int, help='Removes the n most common words from the text.')
    parser.add_argument('--max-freq', type=int, help='Remove words that occur more than freq times in '
                                                     'the input corpus.')
    parser.add_argument('--max-sent-length', type=int, help='Prune most common words such that the max. sentence '
                                                            'length is at most m. If m is too small we only pune '
                                                            'so much that no sentence is empty.')
    return parser


def check_args(args: argparse.Namespace) -> bool:
    none_args = args.strategy == 'none'
    mc_args = args.strategy == 'mc' and args.mc_number
    freq_args = args.strategy == 'freq' and args.max_freq
    sent_length_args = args.strategy == 'sent-length' and args.max_sent_length
    return none_args or mc_args or freq_args or sent_length_args


def word_frequencies(text: List[str]) -> Counter:
    tokens = chain.from_iterable(line.split() for line in text)
    return Counter(tokens)


def sentence_lengths(text: List[str]) -> List[int]:
    return [len(line.split()) for line in text]


def print_word_statistics(text: List[str]) -> None:
    word_freq = word_frequencies(text)
    frequencies = list(word_freq.values())
    line_lengths = sentence_lengths(text)

    print("Minimum sentence length: {}".format(np.min(line_lengths)))
    print("Average sentence length: {}".format(np.mean(line_lengths)))
    print("Maximum sentence length: {}".format(np.max(line_lengths)))
    print("Number of deleted sentences: {}".format(word_freq[DELETED_SENTENCE_TOKEN]))

    print("Average word frequency: {}".format(np.mean(frequencies)))
    print("Maximum word frequency: {}".format(np.max(frequencies)))

    print('10 most common words:')
    for word, freq in word_freq.most_common(10):
        print('{} : {}'.format(word, freq))


def prune_text(text: List[str], filter_set: Set[str]) -> List[str]:
    """
    Prunes list with a specific list of forbidden tokens

    :param text: input text in a form of a list for each sentence
    :param filter_set: set of string-tokens that are to be removed from the text
    :return: pruned list of sentences
    """
    pruned = []
    for line in text:
        pruned_line = [token for token in line.split() if token not in filter_set]
        if not pruned_line:
            pruned_line = [DELETED_SENTENCE_TOKEN]
        pruned.append(' '.join(pruned_line))
    return pruned


def statistics_decorator(headline: str):
    def statistics_decorator(pruner: Callable):
        def wrapped_statistics(text, *args):
            if args:
                print('Before {} (param={}):\n-------------------------'.format(headline, *args))
            else:
                print('Before {}:\n-------------------------'.format(headline))
            print_word_statistics(text)

            pruned = pruner(text, *args)

            print('\nAfter {}:\n-------------------------'.format(headline))
            print_word_statistics(pruned)
            print()
            return pruned

        return wrapped_statistics
    return statistics_decorator


@statistics_decorator('punctuations removal')
def prune_punctuations(text: List[str]) -> List[str]:
    """
    Removes punctuations from text with the help of string.punctuations
    :param text: input text in a form of a list for each sentence
    :return: pruned list of sentences
    """
    return prune_text(text, punctuation)


@statistics_decorator('pruning words by frequency')
def prune_words_by_frequencies(text: List[str], max_frequency: int) -> List[str]:
    """
    Remove words that occur more than certain threshold.

    :param text: input text in a form of a list for each sentence
    :param max_frequency: minimum frequency of a word to be pruned
    :return: pruned list of sentences
    """
    word_freq = word_frequencies(text)
    high_freq_words = {word for word,freq in word_freq.items() if freq > max_frequency}
    return prune_text(text, high_freq_words)


@statistics_decorator('pruning most common words')
def prune_most_common_words(text: List[str], n: int) -> List[str]:
    """
    Removes the first n most-common words from text.

    :param text: input text in a form of a list for each sentence
    :param n: first n most-common words are pruned from the text
    :return: pruned list of sentences
    """
    word_freq = word_frequencies(text)
    most_common = word_freq.most_common(n)

    most_common_words = set(word for word, freq in most_common)
    return prune_text(text, most_common_words)


@statistics_decorator('pruning sentences down to a certain length')
def prune_to_sentence_length(text: List[str], m: int) -> List[str]:
    """
    Prunes most common words until we either obtain a maximum sentence length of m. If m is set too low
    such that sentences are fully pruned, we only remove a number of most-common words such that the minimum sentence
    length is at least 1.

    :param text: input text in a form of a list for each sentence
    :param m: maximum sentence length (might not be possible due to empty sentences that are generated by pruning)
    :return: pruned list of sentences
    """
    # TODO: looks ugly - needs refactoring
    while True:
        indices, length = longest_sentences(text)
        if length <= m:
            return text

        # otherwise we go on pruning
        words_to_prune = most_common_in_sentences(text, indices)
        text = prune_text(text, set(words_to_prune))


def longest_sentences(text: List[str]) -> Tuple[List[int], int]:
    longest_length = max(len(line.split()) for line in text)
    return [index for index, line in enumerate(text) if len(line.split()) == longest_length], longest_length


def most_common_in_sentences(text: List[str], sentence_indices: List[int]) -> Set[str]:
    word_freq = word_frequencies(text)
    to_be_pruned_words = set()

    for index in sentence_indices:
        sent_words = text[index].split()

        if sentence_already_considered(to_be_pruned_words, sent_words):
            continue

        sent_freq = [(word_freq[word], word) for word in text[index].split()]
        sent_freq.sort()
        to_be_pruned_words.add(sent_freq[-1][1])
    return to_be_pruned_words


def sentence_already_considered(pruned_word_set: Set[str], sentence_words: List[str]) -> bool:
    for pruned in pruned_word_set:
        if pruned in sentence_words:
            return True
    return False


if __name__ == '__main__':
    args = create_parser().parse_args()
    if not check_args(args):
        print('Check parameters or type --help for further information.')
        import sys
        sys.exit(1)

    with open(args.input, encoding='UTF-8') as input_data:
        text = input_data.read().splitlines()

    text = prune_punctuations(text)

    if args.strategy == 'none':
        pruned = text
    elif args.strategy == 'mc':
        pruned = prune_most_common_words(text, args.mc_number)
    elif args.strategy == 'freq':
        pruned = prune_words_by_frequencies(text, args.max_freq)
    else:
        pruned = prune_to_sentence_length(text, args.max_sent_length)

    with open(args.output, 'w', encoding='UTF-8') as out:
        for line in pruned:
            out.write(line + '\n')
