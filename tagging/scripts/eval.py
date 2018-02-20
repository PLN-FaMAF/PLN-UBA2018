"""Evaulate a tagger.

Usage:
  eval.py -i <file> [-c]
  eval.py -h | --help

Options:
  -c            Show confusion matrix.
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys
from collections import defaultdict

from ancora import SimpleAncoraCorpusReader


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    # tag
    hits, total = 0, 0
    unk_hits, unk_total = 0, 0
    error_count = defaultdict(lambda: defaultdict(int))
    error_sents = defaultdict(lambda: defaultdict(set))
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)
        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        # global score
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits += sum(hits_sent)
        total += len(sent)
        acc = float(hits) / total * 100

        # score over unknown words
        unk_hits_sent = [hs for w, hs in zip(word_sent, hits_sent) if model.unknown(w)]
        unk_hits += sum(unk_hits_sent)
        unk_total += len(unk_hits_sent)
        unk_acc = float(unk_hits) / unk_total * 100

        # score over known words
        if total == unk_total:
            k_acc = 0.0
        else:
            k_acc = float(hits - unk_hits) / (total - unk_total) * 100

        # confusion matrix
        for t1, t2 in zip(model_tag_sent, gold_tag_sent):
            error_count[t2][t1] += 1
            if t2 != t1:
                # save index of the sentence for error analysis
                error_sents[t2][t1].add(i)

        format_str = '{:3.1f}% ({:2.2f}% / {:2.2f}% / {:2.2f}%)'
        progress(format_str.format(float(i) * 100 / n, acc, k_acc, unk_acc))

    acc = float(hits) / total * 100
    if total == unk_total:
        k_acc = 0.0
    else:
        k_acc = float(hits - unk_hits) / (total - unk_total) * 100
    unk_acc = float(unk_hits) / unk_total * 100

    print('')
    print('Accuracy: {:2.2f}% / {:2.2f}% / {:2.2f}%'.format(acc, k_acc, unk_acc))

    if opts['-c']:
        # print confusion matrix
        print('')

        # basic check
        assert total == sum(sum(d.values()) for d in error_count.values())

        # select most frequent tags
        sorted_error_count = sorted(error_count.keys(),
                                  key=lambda t: -sum(error_count[t].values()))
        entries = sorted_error_count[:10]

        # print table header
        print('g \ m', end='')
        for t in entries:
            print('\t{}'.format(t), end='')
        print('')

        # print table rows
        for t1 in entries:
            print('{}\t'.format(t1), end='')
            for t2 in entries:
                if error_count[t1][t2] > 0:
                    acc = error_count[t1][t2] / total
                    print('{:2.2f}\t'.format(acc * 100), end='')
                else:
                    print('-\t'.format(acc * 100), end='')
            print('')
