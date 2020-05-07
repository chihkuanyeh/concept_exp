from importlib import reload
import sys
from imp import reload
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

def clean_text(text, stop_words, lemmatizer):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def parse_imdb():
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    if sys.version[0] == '2':
        reload(sys)
        sys.setdefaultencoding("utf-8")
    df1 = pd.read_csv('imdb_data/labeledTrainData.tsv', delimiter="\t")
    df1 = df1.drop(['id'], axis=1)
    df2 = pd.read_csv('imdb_data/imdb_master.csv',encoding="latin-1")
    df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)
    df2.columns = ["review","sentiment"]
    df2 = df2[df2.sentiment != 'unsup']
    df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})
    df = pd.concat([df1, df2]).reset_index(drop=True)
    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer()
    df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x, stop_words, lemmatizer))
    print(df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean())
    max_features = 6000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df['Processed_Reviews'])
    list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

    maxlen = 130
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    y = df['sentiment']
    df_test=pd.read_csv("imdb_data/testData.tsv",header=0, delimiter="\t", quoting=3)
    df_test.head()
    df_test["review"]=df_test.review.apply(lambda x: clean_text(x, stop_words, lemmatizer))
    df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
    y_test = df_test["sentiment"]
    list_sentences_test = df_test["review"]
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

    return X_t, X_te, y, y_test