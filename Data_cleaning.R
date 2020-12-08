install.packages("wordcloud")
library(wordcloud)
install.packages("RColorBrewer")
library(RColorBrewer)
install.packages("wordcloud2")
library(wordcloud2)
install.packages("tm")
library(tm)
text <- fake_news$text


#Reading the csv into dataframes
fake_news <- read.csv('Fake.csv')
true_news <- read.csv('True.csv')

news <- read.csv('Final_NEWS.csv')

#Adding F for fake news and T for true news
fake_news['Fake/True'] <- 'F'
true_news['Fake/True'] <- 'T'

#Checking the number of unique values in the subject field
(unique(true_news['subject']))

#Replace the 'politicsNews' in true_news dataframe with 'politics' for consistency
true_news_2 <- true_news
true_news_2$subject <- as.character(true_news_2$subject)
true_news_2[true_news_2=='politicsNews'] <- 'politics'
true_news <- true_news_2


write.csv(fake_news,'Final_fake_news.csv')
write.csv(true_news,'Final_true_news.csv')
write.csv(all_news,'Final_news.csv')


#Combine both the dataframes.
all_news <- rbind(fake_news,true_news)

set.seed(1234)
library(tm)

install.packages("magrittr")
library(magrittr)


#Fake News Title Wordcloud
fake_news_titles <- fake_news$text
docs <- Corpus(VectorSource(fake_news_titles))
docs <- docs %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))
dtm <- TermDocumentMatrix(docs) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)
set.seed(1234) # for reproducibility 
wordcloud(words = df$word, freq = df$freq, min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,colors=brewer.pal(8, "Dark2"))


#True News Title Wordcloud
true_news_titles <- true_news$title
docs <- Corpus(VectorSource(true_news_titles))
docs <- docs %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))
dtm <- TermDocumentMatrix(docs) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)
set.seed(1234) # for reproducibility 
wordcloud(words = df$word, freq = df$freq, min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,colors=brewer.pal(8, "Dark2"))

#All News Titles Wordcloud
all_news_titles <- all_news$text
docs <- Corpus(VectorSource(all_news_titles))
docs <- docs %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))
dtm <- TermDocumentMatrix(docs) 
dtm <- na.omit(dtm)
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)
set.seed(1234) # for reproducibility 
wordcloud(words = df$word, freq = df$freq, min.freq = 1,max.words=100, random.order=FALSE, rot.per=0.35,colors=brewer.pal(8, "Dark2"))

#Twitter hashtags
library(stringr)
library(qdapRegex)




#fake_news_cloud <- wordcloud(words = fake_news$title, min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,colors=brewer.pal(8, "Dark2"))
#true_news_cloud <- wordcloud(words = true_news$title, min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,colors=brewer.pal(8, "Dark2"))
#all_news_cloud <-  wordcloud(words = all_news$title, min.freq = 1,max.words=200, random.order=FALSE, rot.per=0.35,colors=brewer.pal(8, "Dark2"))



