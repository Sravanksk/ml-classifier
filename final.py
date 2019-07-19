from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

# Model Defined Here 

def predict():
	df= pd.read_csv("data.csv", encoding="latin-1")

	# Features and Labels as politics and sports 
	df['label'] = df['class'].map({'politics': 0, 'sport': 1})
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data to process further
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier to predict the classifier of the given input
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	

	if request.method == 'POST':  # take input from the html form as POST request method
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)

	
	return render_template('result.html',prediction = my_prediction) # render the output to result.html file 



if __name__ == '__main__':
	app.run(debug=True)
