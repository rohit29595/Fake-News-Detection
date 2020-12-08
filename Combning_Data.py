import pandas as pd
import nltk
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

true_news = pd.read_csv('real_news-10:21.csv')
true_news_2 = pd.read_csv('True.csv')
true_news_2['date'] = pd.to_datetime(true_news_2['date'])


print(true_news.shape)
quit()

cols = [0,1,2,4,5,7,8,9]
true_news = true_news.drop(true_news.columns[cols], axis=1)
# print(true_news.head())

#true_news.to_csv('realNews20_oct.csv')

true_news['subject'] = 'Other'
true_news_2=true_news_2.rename(columns={'text':'content'})

# print(true_news.columns)
# print(true_news_2.columns)


frames = [true_news,true_news_2]
final_real_news = pd.concat(frames, sort=False)

final_real_news['date'] = pd.to_datetime(final_real_news['date'])
#print(final_real_news.columns)
#print(final_real_news)
final_real_news.to_csv('final_real.csv')


fake_news = pd.read_csv('fake-10:21.csv')
fake_news2 = pd.read_csv('Fake.csv')
#fake_news2['date'] = pd.to_datetime(fake_news2['date'])

#fake_news2['date'] = fake_news2['date'].astype("string")
fake_news2['date'] = pd.to_datetime(fake_news2['date'],errors='coerce')

fake_news['published']=fake_news['published'].astype("string")

#fake_news['date'] = pd.to_datetime(fake_news['published'])

fake_news['date']= pd.to_datetime((fake_news['published'].str[:10]))

print(fake_news.columns)

cols2 = [0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18]
fake_news = fake_news.drop(fake_news.columns[cols2], axis=1)
fake_news['type'] = 'Other'
fake_news=fake_news.rename(columns={'type':'subject'})
fake_news=fake_news.rename(columns={'text':'content'})

fake_news2=fake_news2.rename(columns={'text':'content'})



frames2 = [fake_news,fake_news2]
final_fake_news = pd.concat(frames2, sort=False)
final_fake_news.to_csv('final_fake.csv')

final_fake_news['label'] = 'fake'
final_real_news['label'] = 'true'

frames3 = [final_real_news,final_fake_news]
final_NEWS =  pd.concat(frames3, sort=False)


fake_news['label'] = 'fake'
true_news['label'] = 'true'

fake_news2['label'] = 'fake'
true_news_2['label'] = 'true'


frames4 = [true_news,fake_news]

frames5 = [true_news_2,fake_news2]

mid_f_news = pd.concat(frames4,sort=False)
mid_f_news.to_csv('mid.csv')

tweet_f_news = pd.concat(frames5,sort=False)
tweet_f_news.to_csv('tweets.csv')

plt.figure(figsize= (6, 6))
sns.heatmap(final_NEWS.apply(lambda x: x.factorize()[0]).corr())
plt.show()


final_NEWS.to_csv('final.csv')





#
#
#
# fake_news['date'] = pd.to_datetime(fake_news['published'])
#
# print((fake_news['date']))
#
# fake_news['date'] = fake_news['date'].dt.date
#
# print((fake_news['date']))
