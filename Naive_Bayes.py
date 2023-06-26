#code for multinomial naive bayes

# Step 1 : Libraries to import
import pandas as pd
#import words as words
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
from textblob import TextBlob
from textblob import Word

# Step 1 - check column width
pd.set_option("display.max_colwidth", None)
# Take a tweeter data set with hate speeches and label it as hate or non hate
#hate = racist or sexist = label = 1
# non hate = label = 0
# Choose as many test sets to see how your data looks
train = pd.read_csv(r"C:\Users\18129\Desktop\DATA_HATESPEECH\train.csv")
test = pd.read_csv(r"C:\Users\18129\Desktop\DATA_HATESPEECH\test.csv")
train['label'].value_counts()

#Lets clean train and test separately
#Step 1 Clean and delete the stopwords
sw=stopwords.words("english")
train["tweet"] = train["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
train["tweet"] = train["tweet"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
test["tweet"] = test["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
test["tweet"] = test["tweet"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

#Step - 2 #tokenisation for train and test data tweets and delete the words that occur the least among these words
sil1 = pd.Series(' '.join(train['tweet']).split()).value_counts()[-42000:]
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sil1))
pd.Series(" ".join(train["tweet"]).split()).value_counts()
print(train.shape)

sil2 = pd.Series(' '.join(test['tweet']).split()).value_counts()[-42000:]
test['tweet'] = test['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sil2))
pd.Series(" ".join(test["tweet"]).split()).value_counts()
print(test.shape)
tweet_train = train.tweet
label = train.label
tweet_test = test.tweet
X_train= tweet_train
X_test = tweet_test
Y = label
print(X_train.shape)
print(X_test.shape)
print(Y.shape)

## Plot to show number of racist/sexist words and show number of non hate words
import re
# function to collect hashtags
def hashtag_extract(X_train):
    hashtags = []
    # loop over the words in the tweet

    for i in X_train:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
#extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(X_train
                            [Y == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(X_train
                            [Y== 1])
# unnesting list
HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])
##plot for postiive word distribution
import matplotlib.pyplot as plt
import seaborn as sns
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags
d = d.nlargest(columns = 'Count', n = 10)
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = d, x = "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

## plot for racist/sexist tweet words common
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns = "Count", n = 10)
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = e, x = "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

# Step 3 Clean all the tweets and make it lower case and need to remove floats,alphanumeric
X_train = (X_train.str.strip().str.lower())
X_train = X_train.apply(str)
X_train = X_train.str.strip()
X_train = X_train.str.replace('_', '')
X_train = X_train.str.replace(r"https:\S+", " ")
X_train = X_train.str.replace(r"\n", " ")
X_train= X_train.str.replace(r"#", " ")
X_train = X_train.str.replace(r"@", " ")
X_train = X_train.str.replace(r"!", " ")
X_train = X_train.str.replace(r"0-9", " ")
X_train = X_train.str.replace('[()!?]', ' ')
X_train = X_train.str.replace('\[.*?\]', ' ')
X_train = X_train.str.replace("[^a-z0-9]", " ")

#for test data set too and need to remove floats,alphanumeric
X_test = (X_test.str.strip().str.lower())
X_test = X_test.apply(str)
X_test = X_test.str.strip()
X_test = X_test.str.replace('_', '')
X_test= X_test.str.replace(r"https:\S+", " ")
X_test = X_test.str.replace(r"\n", " ")
X_test= X_test.str.replace(r"#", " ")
X_test = X_test.str.replace(r"@", " ")
X_test = X_test.str.replace(r"!", " ")
X_test = X_test.str.replace(r"0-9", " ")
X_test = X_test.str.replace('[()!?]', ' ')
X_test = X_test.str.replace('\[.*?\]', ' ')
X_test = X_test.str.replace("[^a-z0-9]", " ")

# convert X_train and X_test to DF
X = pd.DataFrame(X_train)
XTest = pd.DataFrame(X_test)
TrainingLabel = pd.DataFrame(Y)

# Use Word Cloud to find most common words in only train data set
import matplotlib.pyplot as plt
 all_words = ' '.join([text for text in X['tweet']])
 from wordcloud import WordCloud
 wordcloud = WordCloud(width = 800, height = 500, random_state = 21,
                      max_font_size = 110).generate(all_words)
 plt.figure(figsize = (10,7))
 plt.imshow(wordcloud, interpolation = 'bilinear')
 plt.axis('off')
 plt.show()

# #We can see most of the words are positive or neutral. With happy, thank, today and love being the most frequent ones. t
# # doesn’t give us any idea about the words associated with the racist/sexist tweets. Hence, we will plot separate wordclouds for both the classes(racist/sexist or not) in our train data.
#
# #Most common words in normal/positive tweets
 normal_words = ' '.join([text for text in
                X['tweet'][Y== 0]])
 wordcloud = WordCloud(width = 800, height = 500, random_state = 21,
                      max_font_size = 110).generate(normal_words)
 plt.figure(figsize=(10, 7))
 plt.imshow(wordcloud, interpolation="bilinear")
 plt.axis('off')
 plt.show()

# #Racist/sexist tweets
 negative_words = ' '.join([text for text in
                 X['tweet'][Y == 1]])

 wordcloud = WordCloud(width = 800, height = 500, random_state = 21,
                      max_font_size = 110).generate(negative_words)
 plt.figure(figsize = (10, 7))
 plt.imshow(wordcloud, interpolation = 'bilinear')
 plt.axis('off')
 plt.show()

# classifier work
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#Split the train data set to train and test set to create model
X_Train, X_Test, TrainingLabel_train, TrainingLabel_test = train_test_split(
                 X, TrainingLabel, random_state = 42, test_size = 0.3)

## Problem statement
# XTest Test data fow which we have to find the labels
#X- Train data
#TrainingLabel
#Predict labels of XTest
# We need to split the existing train data set to get a model which can be used to predict Test data

#Method 1
## Feature extraction uses two methods BOW and TF-IDF
#Term frequency–inverse document frequency (TF-IDF), takes into account not just the occurrence of a word in a single document (or tweet) but in the entire corpus. TF-IDF works by penalizing the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.

# TF = (number of appearance of a term t)/(number of total terms in the document)
# IDF = log(N/n), where N is the number of documents and n is the number of documents a term t has appeared in.
# TF-IDF = TF * IDF
#Precision = TP/TP+FP

# Recall = TP/TP+FN
#
# F1 Score = 2(Recall Precision) / (Recall + Precision)
#
# Where TP - true positive, FP - false positive, TN - true negative, FN - false negative.


## Model 1
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer =  TfidfVectorizer(max_df = 0.9, min_df = 2,
                                  max_features = 1000, stop_words = 'english')


# TF-IDF feature matrix

tfidf_Train = tfidf_vectorizer.fit_transform(X_Train['tweet'])
tfidf_Test = tfidf_vectorizer.transform(X_Test['tweet'])
TFIDFTestDataSet = tfidf_vectorizer.transform(XTest['tweet'])
TFIDFTrainDataSet = tfidf_vectorizer.transform(X['tweet'])

#Case 2 -Method 2
#Bag-of-words features

# Bag-of-Words is a method to convert text into numerical features. Bag-of-Words features can be easily created using sklearn’s CountVectorizer function. We will set the parameter
# max_features = 1000 to select only top 1000 terms ordered by term frequency across the corpus.

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df = 0.9, min_df = 2,
                                max_features = 1000, stop_words = 'english')

# bag-of-words feature matrix

bow_Train = bow_vectorizer.fit_transform(X_Train['tweet'])
bow_Test =bow_vectorizer.transform(X_Test['tweet'])
bowTestDataSet = bow_vectorizer.transform(XTest['tweet'])
bowTrainDataSet =  bow_vectorizer.transform(X['tweet'])


#Method 3 - Generic Method
vec = CountVectorizer(stop_words='english')

XTrain_generic = vec.fit_transform(X_Train['tweet'])
XTest_generic= vec.transform(X_Test['tweet'])
TestDataSet_generic = vec.transform(XTest['tweet'])
TrainDataSet_generic = vec.transform(X['tweet'])


# Lets find what is the prediction for the labels of our test data set using our train data set


# Case1 - For Bag of words
model = MultinomialNB()
Fit_bow = model.fit(bow_Train, TrainingLabel_train)
Score_bow = model.score(bow_Test,TrainingLabel_test)
prediction_bow = model.predict(bow_Test)
# Test Data Set Prediction accuracy using BOW
prediction_bow_TestSet = model.predict(bowTestDataSet )
print("The  accuracy score for using only the bag-of-words features is : {}".format(Score_bow))
#Score1 = model.score(bowTestDataSet,prediction_bow_TestSet)

# F1 SCORE BOW
import numpy as np

prediction =model.predict_proba(bow_Test)
prediction_int = prediction[:,1] >= 0.3 #if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print(classification_report(TrainingLabel_test,prediction_int))
accuracy_score(TrainingLabel_test,prediction_int)
print("The f1-score for using only the Bag of Words features is : {}".format(classification_report(TrainingLabel_test,prediction_int)))


# Case 2 - For TFIDF
model = MultinomialNB()
Fit_TD = model.fit(tfidf_Train , TrainingLabel_train)
Score_Td = model.score(tfidf_Test,TrainingLabel_test)
prediction_td = model.predict(tfidf_Train  )
prediction_td_TestDataSet = model.predict(TFIDFTestDataSet )
print("The accuracy score for using only the TFIDF features is : {}".format(Score_Td))

# F1 SCORE TFIDF
import numpy as np
# # make prediction on validation set
prediction_F1_TFIDF =model.predict_proba(tfidf_Test)
prediction_int_TFIDF = prediction[:,1] >= 0.3 #if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int_TFIDF = prediction_int_TFIDF.astype(np.int)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print(classification_report(TrainingLabel_test,prediction_int_TFIDF))
accuracy_score(TrainingLabel_test,prediction_int_TFIDF)
print("The f1-score for using only the TFIDF features is : {}".format(classification_report(TrainingLabel_test,prediction_int_TFIDF)))

# Case 3 - generic supervision use supervised classifier to check confidence score of the test data
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
Fit = nb.fit(XTrain_generic, TrainingLabel_train)
Score = nb.score(XTest_generic ,TrainingLabel_test)
prediction_generic = nb.predict_proba(XTest_generic)
prediction_int_generic = prediction_generic[:,1] >= 0.3 #if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int_generic = prediction_int_generic.astype(np.int)
prediction_Test_generic = nb.predict(TestDataSet_generic)
print("The accuracy score for using only the mutinomial model  features is : {}".format(Score))


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print(classification_report(TrainingLabel_test,prediction_int_generic))
accuracy_score(TrainingLabel_test,prediction_int_generic)
print("The f1-score for not using TFIDF and Bag of words features is : {}".format(classification_report(TrainingLabel_test,prediction_int_generic)))

# Validation of the model using test data sets to check how the model performs


Test_new_bow = pd.DataFrame([test['tweet'], prediction_bow_TestSet])
Test_new_TFIDF = pd.DataFrame([test['tweet'], prediction_td_TestDataSet])
Test_new_none = pd.DataFrame([test['tweet'], prediction_Test_generic])

# Accuracy of the train data set used to create the model
PercPosTweet_Train= (sum(TrainingLabel['label']==0)/len(train['tweet'])*100)
PercNegTweet_Train= (sum(TrainingLabel['label']==1)/len(train['tweet'])*100)

# so basically our measurement is which model gives better search of hate tweets and why ?


# pie chart for trained data set
arr1= np.array([PercPosTweet_Train , PercNegTweet_Train  ])
mylabels1 = ["Pos=PercPosTweet_Train ", "Neg=PercNegTweet_Train", ]

plt.figure(6)
plt.pie(arr1, labels = mylabels1,autopct='%1.1f%%')
plt.title( 'Hate/NonHate distribution for the Trained Data set used to create the model- Actual Value ')



# you can use any tweet set to check prediction
# plot label prediction accuracy from the test data set using the model created

PercPosTweets_bow= (sum(Test_new_bow.T ['Unnamed 0']==0)/len(Test_new_bow.T['tweet'])*100)
PercNegTweets_bow= (sum(Test_new_bow.T ['Unnamed 0']==1)/len(Test_new_bow.T['tweet'])*100)

PercPosTweets_TFIDF= (sum(Test_new_TFIDF.T ['Unnamed 0']==0)/len(Test_new_TFIDF.T['tweet'])*100)
PercNegTweets_TFIDF= (sum(Test_new_TFIDF.T ['Unnamed 0']==1)/len(Test_new_TFIDF.T['tweet'])*100)


PercPosTweets_generic= (sum(Test_new_none.T ['Unnamed 0']==0)/len(Test_new_none.T['tweet'])*100)
PercNegTweets_generic= (sum(Test_new_none.T ['Unnamed 0']==1)/len(Test_new_none.T['tweet'])*100)

# pie chart for predicted data set

arr4= np.array([PercPosTweets_bow , PercNegTweets_bow  ])
mylabels4 = ["Pos=PercPosTweets_bow ", "Neg=PercNegTweets_bow" ]
arr5= np.array([PercPosTweets_TFIDF , PercNegTweets_TFIDF ])
mylabels5 = ["Pos=PercPosTweets_TFIDF ", "Neg=PercNegTweets_TFIDF"]
arr6= np.array([PercPosTweets_generic, PercNegTweets_generic  ])
mylabels6 = ["Pos=PercPosTweet_generic ", "Neg=PercNegTweets_generic" ]
plt.figure(9)
plt.pie(arr4, labels = mylabels4,autopct='%1.1f%%')
plt.title( 'Hate/NonHate distribution for the test data set using the labels predicted by the classifier model Bag of Words NB ')
plt.figure(10)
plt.pie(arr5, labels = mylabels5 ,autopct='%1.1f%%')
plt.title( 'Hate/NonHate distribution for the test data set using the labels predicted by the classifier model TFIDF NB')
plt.figure(11)
plt.pie(arr6, labels = mylabels6,autopct='%1.1f%%')
plt.title('Hate/NonHate distribution for the test data set using the labels predicted by the classifier model  NB ')

#plot model confidence scores using NB