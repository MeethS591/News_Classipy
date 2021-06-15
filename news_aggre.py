import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from newsapi import NewsApiClient

category_list = ['business', 'entertainment', 'health', 'technology']
newsapi = NewsApiClient(api_key='api_key')


top_headlines = newsapi.get_top_headlines(
                                          language='en',
                                          country='in')

docs_new = top_headlines
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("model_pickel/count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("model_pickel/tfidf.pkl","rb"))
loaded_model = pickle.load(open("model_pickel/model.pkl","rb"))
#print(docs_new)
for news in docs_new['articles']:
    news=[news["title"]]
    print(news)
    X_new_counts = loaded_vec.transform(news)
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)
    print(category_list[int(predicted)])
