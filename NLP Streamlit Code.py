from pyexpat import features
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
#import tika
#tika.initVM()
#from tika import parser
from pdfminer.high_level import extract_text


import nltk
import re
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()

stopwords_extra = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
stopwords_all = stopwords.words("english")
stopwords_all.extend(stopwords_extra)

from nltk.stem.wordnet import WordNetLemmatizer
lm = WordNetLemmatizer()
from matplotlib.pyplot import imshow
from sklearn.feature_extraction.text import TfidfVectorizer


with open('model_booksummary_6.pkl', 'rb') as f6:
    mp_6 = pickle.load(f6)
with open('model_booksummary_5.pkl', 'rb') as f5:
    mp_5 = pickle.load(f5)
with open('model_booksummary_4.pkl', 'rb') as f4:
    mp_4 = pickle.load(f4)
with open('model_booksummary_3.pkl', 'rb') as f3:
    mp_3 = pickle.load(f3)
with open('model_booksummary_2.pkl', 'rb') as f2:
    mp_2 = pickle.load(f2)
with open('model_booksummary_1.pkl', 'rb') as f1:
    mp_1 = pickle.load(f1)
with open('model_booksummary_0_5.pkl', 'rb') as f_05:
    mp_0_5 = pickle.load(f_05)


st.title("Book Summary Extraction")

book = st.sidebar.file_uploader("Upload PDF file")
if book != None:
    #from tika import parser 
    #raw = parser.from_file(book)
    #book_content = (raw['content'])
    book_content = extract_text(book)
    
    
    sentence=nltk.sent_tokenize(book_content)
    
    list1=[]

    for i in range(0, len(sentence)):
        review=re.sub('[^a-zA-Z]',' ',sentence[i]).lower().split()
        review=[lm.lemmatize(word) for word in review if not word in stopwords_all]
        review=' '.join(review)
        list1.append(review)
        
    vectorizer_general=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1))
    data_general = vectorizer_general.fit_transform(list1).toarray()
    max_feat_exact = (data_general.shape)[1]
    
    max_feat = 1000
    model = mp_6
    
    if max_feat_exact >= 6000:
        max_feat = 6000
        model = mp_6
    elif max_feat_exact >= 5000 and max_feat_exact < 6000:
        max_feat = 5000
        model = mp_5
    elif max_feat_exact >= 4000 and max_feat_exact < 5000:
        max_feat = 4000
        model = mp_4
    elif max_feat_exact >= 3000 and max_feat_exact < 4000:
        max_feat = 3000
        model = mp_3
    elif max_feat_exact >= 2000 and max_feat_exact < 3000:
        max_feat = 2000
        model = mp_2
    elif max_feat_exact >= 1000 and max_feat_exact < 2000:
        max_feat = 1000
        model = mp_1
    elif max_feat_exact >= 500 and max_feat_exact < 1000:
        max_feat = 500
        model = mp_0_5
    else:
        None
        
    vectorizer_7000=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1), max_features=max_feat)
    data_7000 = vectorizer_7000.fit_transform(list1).toarray()
    
    y_pred = model.predict(data_7000)
    
    negative = pd.DataFrame(y_pred).value_counts()[0]
    neutral  = pd.DataFrame(y_pred).value_counts()[1]
    positive = pd.DataFrame(y_pred).value_counts()[2]
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Sentences", np.round(positive/(negative+positive+neutral),2)*100, "")
    col2.metric("Negative Sentences", np.round(negative/(negative+positive+neutral),3)*100, "")
    col3.metric("neutral Sentences", np.round(neutral/(negative+positive+neutral),3)*100, "")
    
#    chart_data = pd.DataFrame(
#        {"negative":[negative],
#        "positive":[positive],
#        "neutral":[neutral]})

#    st.bar_chart(chart_data)

    if positive > negative:
        st.info('This is Positive sentiment book')
        st.balloons()
    elif negative > positive:
        st.warning('This is Negative sentiment book')

        

    exp_vals = [positive,negative,neutral]
    exp_labels = ["Positive","Negative","neutral"]
    
    fig2 = plt.figure(figsize = (10, 5))
    plt.axis("equal")
    plt.pie(exp_vals,labels=exp_labels, shadow=False, autopct='%2.1f%%',radius=1.2,explode=[0,0,0],counterclock=True, startangle=45)
    st.pyplot(fig2)


    def bar_chart():
        #Creating the dataset
        data = {'Positive Sentences':positive, 'Negative Sentences':negative, 'neutral Sentences':neutral}
        Courses = list(data.keys())
        values = list(data.values())

        fig = plt.figure(figsize = (10, 5))

        plt.bar(Courses, values)
        plt.xlabel("Sentiments")
        plt.ylabel("")
        plt.title("")
        st.pyplot(fig)
    
#    bar_chart()


    
    
    
    






























