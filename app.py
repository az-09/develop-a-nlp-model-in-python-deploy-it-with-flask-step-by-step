from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    cv_Vocabulary = open('./model/CV_vocabulary.pkl','rb')
    cv = joblib.load(cv_Vocabulary) 
    NB_spam_model = open('./model/NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vec = cv.transform(data).toarray()
        my_prediction = clf.predict(vec)
    return render_template('index.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)