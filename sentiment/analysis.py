"""Model analysis tools."""


def print_maxent_features(vect, clf, n=5):
	"""
	Most relevant features for each class (logistic regression).

	vect -- vectorizer (count or tf-idf)
	clf -- LogisticRegression classifier
	n -- number of features to show
	"""
	C = clf.coef_
	A = clf.coef_.argsort()
	features = vect.get_feature_names()
	for i, label in enumerate(clf.classes_):
		print('{}:'.format(label))
		print('\t{} ({})'.format(
			' '.join([features[j] for j in A[i, :5]]),
			C[i, A[i, :n]]))
		print('\t{} ({})'.format(
			' '.join([features[j] for j in A[i, -5:]]),
			C[i, A[i, -n:]]))
