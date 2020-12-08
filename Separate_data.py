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
import timeit
from sklearn.ensemble import RandomForestClassifier


t_news = pd.read_csv('real_news-10:21.csv')
f_news = pd.read_csv('fake-10:21.csv')


cols = [0,1,2,6,7,8,9]
t_news = t_news.drop(t_news.columns[cols], axis=1)
t_news['label'] = 'true'





f_news = pd.read_csv('fake-10:21.csv')
cols2 = [0,1,3,6,7,9,10,11,12,13,14,15,16,17,18,19]
f_news = f_news.drop(f_news.columns[cols2], axis=1)
f_news['label'] = 'fake'


t_news=t_news.rename(columns={'text':'content'})

f_news=f_news.rename(columns={'site_url':'publication'})



# print(t_news.columns)
# print(f_news.columns)

# t_news['date'] = pd.to_datetime(t_news['date'])

# t_news['label'] = 'true'


# f_news = pd.read_csv('fake-10:21.csv')
# f_news['date'] = pd.to_datetime(f_news['date'],errors='coerce')
# f_news['label'] = 'fake'


fr = [t_news,f_news]
fin_f_news = pd.concat(fr, sort=False)
fin_f_news.to_csv('fN_news.csv')






fin_f_news=fin_f_news.rename(columns={'text':'content'})


x_train,x_test,y_train,y_test = train_test_split(fin_f_news['content'], fin_f_news.label, test_size=0.2, random_state=1)

v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
x = v.fit_transform(fin_f_news.ravel())  ## Even astype(str) would work

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


