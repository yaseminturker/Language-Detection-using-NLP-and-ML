# importing libraries
from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

application = Flask(__name__)
@application.route('/')

def home():
    return render_template("main.html")

@application.route("/predict", methods = ["POST"])

def predict():
    # loading the dataset
    data = pd.read_csv("Language_Detection.csv")
    y = data["Language"]

    # label encoding
    y = le.fit_transform(y)

    #loading the model and cv
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))

    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        # preprocessing the text
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]
        # creating the vector
        vect = cv.transform(dat).toarray()
        # prediction
        my_pred = model.predict(vect)
        my_pred = le.inverse_transform(my_pred)

    return render_template("main.html", pred="This word/sentence contains {} word(s).".format(my_pred[0]))


if __name__ =="__main__":
    application.run(debug=True)