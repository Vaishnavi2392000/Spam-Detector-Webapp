import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem .porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask,render_template,request



# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'spam-sms-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cs = pickle.load(open('cs-transform.pkl','rb'))

app = Flask(__name__, static_url_path='/static') 


@app.route('/',methods=['GET','POST'])
def home():
	return render_template('home.html')


@app.route('/spam',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        with open('model_pickle','rb') as f:
            mp=pickle.load(f)
        text = request.form['text']
        data = [text]
        vect = cs.transform(data).toarray()
        y_pred = mp.predict(vect)
        return render_template('result.html', y_pred=y_pred)
    else:
        return render_template('form.html')



if __name__ == '__main__':
	app.run(debug=True)

    		



