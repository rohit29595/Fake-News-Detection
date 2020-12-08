import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix


#Imprting Data
# true = pd.read_csv('Fake.csv')
# fake = pd.read_csv('True.csv')

news = pd.read_csv('news.csv')
news = news.drop('Unnamed: 0',axis=1)

news2 = pd.read_csv('tweets.csv')
news2 = news2.drop('Unnamed: 0',axis=1)

news3 = pd.read_csv('mid.csv')
news3 = news3.drop('Unnamed: 0',axis=1)

print(news.columns)
print(news2.columns)
print(news3.columns)



# fake['target'] = 'fake'
# true['target'] = 'true'
# #News dataset
# news = pd.concat([fake, true]).reset_index(drop = True)


news = news.replace(to_replace='politicsNews',value='politics')

# print(news.groupby(['subject'])['content'].count())
# news.groupby(['subject'])['content'].count().plot(kind='bar')
# plt.show()

# print(pd.Series(' '.join((news.content).astype(str)).split()).value_counts()[:3])



#
# news['content'] = news['content'].astype('U')
# news['title'] = news['title'].astype('U')

print('Accuracies of Mid dataset :')

news3['content'] = news3['content'].astype('U')


# print(news['content'].dtype)
x_train,x_test,y_train,y_test = train_test_split(news3.content, news3.label, test_size=0.2, random_state=1)

v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
x = v.fit_transform(news3['content'].values.astype('U'))  ## Even astype(str) would work

pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])
# x = pipe1.fit_transform(news['text'].values.astype('U').values)

model_lr = pipe1.fit(x_train, y_train)
lr_pred = model_lr.predict(x_test)
#
print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(y_test, lr_pred)*100,2)))




pipe2 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
model = pipe2.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Accuracy of Random Forest Classifier: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))



pipe3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', MultinomialNB())])

model_nb = pipe3.fit(x_train, y_train)
nb_pred = model_nb.predict(x_test)

print("Accuracy of Naive Bayes Classifier: {}%".format(round(accuracy_score(y_test, nb_pred)*100,2)))


print('--------------------------------------------------------')


print('Accuracies of NEWS dataset :')

news['text'] = news['text'].astype('U')

#
# # print(news['content'].dtype)
# x_train,x_test,y_train,y_test = train_test_split(news.text, news.label, test_size=0.2, random_state=1)
#
# v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
# x = v.fit_transform(news['text'].values.astype('U'))  ## Even astype(str) would work
#
# pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])
# # x = pipe1.fit_transform(news['text'].values.astype('U').values)
#
# model_lr = pipe1.fit(x_train, y_train)
# lr_pred = model_lr.predict(x_test)
# #
# print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(y_test, lr_pred)*100,2)))
#
#
#
#
# pipe2 = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
# model = pipe2.fit(x_train, y_train)
# prediction = model.predict(x_test)
# print("Accuracy of Random Forest Classifier: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#


pipe3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', MultinomialNB())])

model_nb = pipe3.fit(x_train, y_train)
nb_pred = model_nb.predict(x_test)

print("Accuracy of Naive Bayes Classifier: {}%".format(round(accuracy_score(y_test, nb_pred)*100,2)))


print(multilabel_confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


# CLean the text - Removing punctuation and stopwords

# def clean_text(s):
#
#     without_punc = [c for c in s if c not in string.punctuation]
#     without_punc = ''.join(without_punc)
#     clean_str = [word for word in without_punc.split() if word.lower() not in stopwords.words('english')]
#
#     return clean_str
#
# news['content'] = news['content'].apply(clean_text)
#
# print('Done')
