"""Draw a learning curve for a Sentiment Analysis model.

Usage:
  curve.py [-m <model>] [-c <clf>]
  curve.py -h | --help

Options:
  -m <model>    Model to use [default: basemf]:
                  basemf: Most frequent sentiment
                  clf: Machine Learning Classifier
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  -h --help     Show this screen.
"""
from docopt import docopt

from sentiment.tass import InterTASSReader, GeneralTASSReader
from sentiment.baselines import MostFrequent
from sentiment.classifier import SentimentClassifier
from sentiment.evaluator import Evaluator


models = {
    'basemf': MostFrequent,
    'clf': SentimentClassifier,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load training corpus
    reader1 = InterTASSReader('TASS/InterTASS/tw_faces4tassTrain1000rc.xml')
    X1, y1 = list(reader1.X()), list(reader1.y())
    reader2 = GeneralTASSReader('TASS/GeneralTASS/general-tweets-train-tagged.xml', simple=True)
    X2, y2 = list(reader2.X()), list(reader2.y())
    X, y = X1 + X2, y1 + y2

    # load development corpus (for evaluation)
    reader = InterTASSReader('TASS/InterTASS/TASS2017_T1_development.xml')
    Xdev, y_true = list(reader.X()), list(reader.y())

    # create model and evaluator instances
    # train model
    model_type = opts['-m']
    if model_type == 'clf':
        model = models[model_type](clf=opts['-c'])
    else:
        model = models[model_type]()  # baseline
    evaluator = Evaluator()

    N = len(X)
    for i in reversed(range(8)):
        n = int(N / 2**i)
        this_X = X[:n]
        this_y = y[:n]

        # train, test and evaluate
        model.fit(this_X, this_y)
        y_pred = model.predict(Xdev)
        evaluator.evaluate(y_true, y_pred)

        # print this data point:
        acc = evaluator.accuracy()
        f1 = evaluator.macro_f1()
        print('n={}, acc={:2.2f}, f1={:2.2f}'.format(n, acc, f1))
