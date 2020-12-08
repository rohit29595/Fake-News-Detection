#Importing Required Libraries
from nltk import tokenize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import multilabel_confusion_matrix


#Importing Data
news = pd.read_csv('mid.csv')
news = news.drop('Unnamed: 0',axis=1)
news['content'] = news['content'].astype('U')



news = news.drop('date',axis=1)
news = news.drop('subject',axis=1)

news2 = pd.read_csv('news.csv')
news2 = news2.drop('Unnamed: 0',axis=1)
news2=news2.rename(columns={'text':'content'})

news3 = pd.read_csv('tweets.csv')
news3 = news3.drop('Unnamed: 0',axis=1)


frames3 = [news,news3]
news =  pd.concat(frames3, sort=False)

# news.to_csv('REAL-FINAL.csv')

# stemmer = SnowballStemmer("english")
# news['unstemmed'] = news['content'].str.split()
#
# news['stemmed'] = news['unstemmed'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
# news = news.drop(columns=['unstemmed'])
#
# # print(news.head)
# #
# #
# news['stemmed'] = news['stemmed'].astype('U')
# print(news['stemmed'].dtype)
x_train,x_test,y_train,y_test = train_test_split(news['content'], news.label, test_size=0.2, random_state=1)

v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
x = v.fit_transform(news['content'].values.astype('U'))  ## Even astype(str) would work

# pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression(max_iter=5000))])
# # x = pipe1.fit_transform(news['text'].values.astype('U').values)
#
# model_lr = pipe1.fit(x_train, y_train)
# lr_pred = model_lr.predict(x_test)
# #
# print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(y_test, lr_pred)*100,2)))
# #
# pipe2 = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
# model = pipe2.fit(x_train, y_train)
# prediction = model.predict(x_test)
# print("Accuracy of Random Forest Classifier: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

pipe3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', MultinomialNB())])

model_nb = pipe3.fit(x_train, y_train)
nb_pred = model_nb.predict(x_test)

print("Accuracy of Naive Bayes Classifier: {}%".format(round(accuracy_score(y_test, nb_pred)*100,2)))

print(multilabel_confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))





# token_space = tokenize.WhitespaceTokenizer()
# def counter(text, column_text, quantity):
#     all_words = ' '.join([text for text in text[column_text]])
#     token_phrase = token_space.tokenize(all_words)
#     frequency = nltk.FreqDist(token_phrase)
#     df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
#                                    "Frequency": list(frequency.values())})
#     df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
#     plt.figure(figsize=(12,8))
#     ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
#     ax.set(ylabel = "Count")
#     plt.xticks(rotation='vertical')
#     plt.show()
#
# counter(news[news['label'] == 'fake'], 'content', 20)
