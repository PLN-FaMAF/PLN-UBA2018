from xml.etree import ElementTree


class GeneralTASSReader:

    def __init__(self, filename, res_filename=None, simple=False):
        self.filename = filename
        self.res_filename = res_filename
        self.simple = simple

        self.root = ElementTree.parse(filename).getroot()
        self.smapdict = smapdict = {}
        if simple:
            smapdict['P+'] = 'P'
            smapdict['N+'] = 'N'

    def tweets(self):
        """Iterator over the tweets."""

        def smap(sentiment):
            return self.smapdict.get(sentiment, sentiment)

        for tweet_el in self.root:
            assert len(tweet_el) == 7
            attrs = ['tweetid', 'user', 'content', 'date', 'lang']
            tweet = {}
            for attr in attrs:
                tweet[attr] = tweet_el.find(attr).text

            # general sentiment
            polarity_el = tweet_el.find('sentiments')[0]
            tweet['sentiment'] = {
                'value': smap(polarity_el.find('value').text),
                'type': polarity_el.find('type').text
            }

            # entity sentiments
            tweet['sentiments'] = []
            for polarity_el in tweet_el.find('sentiments')[1:]:
                polarity = {
                    'entity': polarity_el.find('entity').text,
                    'value': smap(polarity_el.find('value').text),
                    'type': polarity_el.find('type').text
                }
                tweet['sentiments'].append(polarity)

            # now the topics
            tweet['topics'] = []
            for topic_el in tweet_el.find('topics'):
                tweet['topics'].append(topic_el.text)

            yield tweet

    def X(self):
        """Iterator over the tweet contents."""

        for tweet_el in self.root:
            # assert len(tweet_el) in [5, 7]
            content = tweet_el.find('content').text or ''
            yield content

    def y(self):
        """Iterator over the tweet polarities."""
        def smap(sentiment):
            return self.smapdict.get(sentiment, sentiment)

        if self.res_filename is None:
            # development dataset
            for tweet_el in self.root:
                assert len(tweet_el) == 7
                # general sentiment
                polarity_el = tweet_el.find('sentiments')[0]
                sentiment = smap(polarity_el.find('value').text)
                yield sentiment
        else:
            # test dataset.
            # tweets in the qrel file must be in the same order as in the XML
            with open(self.res_filename, 'r') as f:
                for line in f:
                    sentiment = line.split()[-1]
                    yield sentiment


class InterTASSReader:

    def __init__(self, filename, res_filename=None):
        self.filename = filename
        self.res_filename = res_filename
        self.root = ElementTree.parse(filename).getroot()

    def tweets(self):
        """Iterator over the tweets."""
        for tweet_el in self.root:
            assert len(tweet_el) == 6
            attrs = ['tweetid', 'user', 'content', 'date', 'lang']
            tweet = {}
            for attr in attrs:
                tweet[attr] = tweet_el.find(attr).text
            # now the sentiment
            tweet['sentiment'] = tweet_el.find('sentiment')[0][0].text

            yield tweet

    def X(self):
        """Iterator over the tweet contents."""
        for tweet_el in self.root:
            assert len(tweet_el) == 6
            content = tweet_el.find('content').text
            yield content

    def y(self):
        """Iterator over the tweet polarities."""
        if self.res_filename is None:
            # development dataset
            for tweet_el in self.root:
                assert len(tweet_el) == 6
                sentiment = tweet_el.find('sentiment')[0][0].text
                yield sentiment
        else:
            # test dataset
            with open(self.res_filename, 'r') as f:
                for line in f:
                    sentiment = line.split()[-1]
                    yield sentiment
