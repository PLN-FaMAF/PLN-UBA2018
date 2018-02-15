# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.scripts.stats import POSStats


class TestStats(TestCase):

    def setUp(self):
        self.tagged_sents = [
            list(zip('el gato come pescado .'.split(),
                 'D N V N P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]

    def test_words(self):
        stats = POSStats(self.tagged_sents)

        words = {'el', 'gato', 'come', 'pescado', 'la', 'gata', 'salmón', '.'}

        self.assertEqual(set(stats.words()), words)
