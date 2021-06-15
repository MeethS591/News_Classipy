import pandas as pd
news = pd.read_csv("news-aggregator-dataset/uci-news-aggregator.csv")
#print(str(news))
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])
print(y[:5])

categories = news['CATEGORY']
titles = news['TITLE']
print(type(news))
N = len(titles)
print('Number of news',N)

labels = list(set(categories))
print('possible categories',labels)

for l in labels:
    print('number of ',l,' news',len(news.loc[news['CATEGORY'] == l]))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
ncategories = encoder.fit_transform(categories)
Ntrain = int(N * 0.7)

from sklearn.utils import shuffle
titles, ncategories = shuffle(titles, ncategories, random_state=0)

X_train = titles[:Ntrain]
print('X_train.shape',X_train.shape)
y_train = ncategories[:Ntrain]
print('y_train.shape',y_train.shape)

X_test = titles[Ntrain:]
print('X_test.shape',X_test.shape)
y_test = ncategories[Ntrain:]
print('y_test.shape',y_test.shape)
print(y_test)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

print('Training...')
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
pickle.dump(count_vect.vocabulary_, open("models_pickle/count_vector.pkl","wb"))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
pickle.dump(tfidf_transformer, open("models_pickle/tfidf.pkl","wb"))


text_clf=MultinomialNB().fit(X_train_tfidf, y_train)
pickle.dump(text_clf, open("models_pickle/model.pkl", "wb"))
"""
from sklearn import metricss

print('accuracy_score',metrics.accuracy_score(y_test,predicted))
print('Reporting...')

print(metrics.classification_report(y_test, predicted, target_names=labels))"""
