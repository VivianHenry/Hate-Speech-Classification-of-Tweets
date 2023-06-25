<h1>Hate Speech Classification of Tweets</h1>

<h2>Introduction</h2>

The aim of this project is to combine elements of social media mining, text mining, and classification into a wholistic project that has real world applications. In this pursuit, a Twitter sentiment analysis will be conducted to detect hate speech in tweets. Hate speech here includes those tweets that have a trace of racist or sexist sentiment associated with them.

An overview of the project goals is given below:
<ul>
  <li>Analyze the sentiment of tweets and detect hate speech.</li>
  <li>Implement 3 different classification algorithms (Naïve Bayes, SVM, and Logistic Regression). to predict the overall fit, score, and confidence of the prediction model.</li>
  <li>Compare the 3 models via overall fit, F1-score, and confidence.</li>
</ul>

<h2>Methodology</h2>

<ol>
  <li>Data Collection: The dataset is from Kaggle. These are in the .csv format.</li>
  <li>Preprocessing: The dataset undergoes data wrangling.</li>
  <li>Sentiment Analysis: Sentiment Analysis is employed to decide the polarity of the tweet. This process predicts if the sentiment of a tweet is positive, negative, or neutral. This is a predecessor to predicting whether the Tweet is racist/sexist.</li>
  <li>Train and Test Data creation: Dataset is split into training and test data.</li>
  <li>Vectorization of Tweets: Each feature in the tweet is converted from text to a numeric array using vectorization.</li>
  <li>Classifier Algorithms: Training dataset is used to create the classification models.</li>
  <li>Testing: In order to quantify their performance, the accuracy, f1-score, fit, and confidence of each model is computed.</li>
</ol>

![Screenshot 2023-06-25 at 7 51 06 PM](https://github.com/VivianHenry/Hate-Speech-Classification-of-Tweets/assets/67223688/a6e6852a-5fb9-4c7b-869a-05a144778090)

<img width="800" alt="image" src="https://github.com/VivianHenry/Hate-Speech-Classification-of-Tweets/assets/67223688/a6e6852a-5fb9-4c7b-869a-05a144778090">

