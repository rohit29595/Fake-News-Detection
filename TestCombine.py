import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


true_news_2 = pd.read_csv('True.csv')
true_news_2['date'] = pd.to_datetime(true_news_2['date'])
true_news_2=true_news_2.rename(columns={'text':'content'})
fake_news2 = pd.read_csv('Fake.csv')
fake_news2['date'] = pd.to_datetime(fake_news2['date'],errors='coerce')
fake_news2=fake_news2.rename(columns={'text':'content'})

fake_news2['label'] = 'fake'
true_news_2['label'] = 'true'

frames = [fake_news2,true_news_2]
final = pd.concat(frames, sort=False)

plt.figure(figsize= (6, 6))
sns.heatmap(final.apply(lambda x: x.factorize()[0]).corr())
plt.show()

final.to_csv('final.csv')


