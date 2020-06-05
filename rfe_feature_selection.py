# evaluate RFE for classification
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot
import altair as alt
import seaborn as sn

@st.cache
def loadData():
	df = pd.read_csv("stock_data.csv")
	X = df.iloc[:, 0:100]
	y = df.iloc[:,-1]
	return df,X,y

# Basic splitting required for all the models.  
def split(X,y):
	# 1. Splitting X,y into Train & Test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
	return X_train, X_test, y_train, y_test

# get a list of models to evaluate
# @st.cache(suppress_st_warning=True)
def get_models():
	models = dict()
	for i in range(2, 100):
		rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		model = DecisionTreeClassifier()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

# evaluate a give model using cross-validation
@st.cache(suppress_st_warning=True)
def evaluate_model(model,X_train,y_train):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train,y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

def visualize_data(df, x_axis, y_axis):
	graph = alt.Chart(df).mark_circle(size=60).encode(
	x=x_axis,
	y=y_axis).configure_axis(
    grid=False).interactive()
	st.write(graph)

def main():
	st.title("Using Streamlit Apps - Feature Selection for Classification Problems using various Machine Learning Classification Algorithms")
	df,X,y = loadData()

	page = st.sidebar.selectbox("Choose a page",["Homepage", "Exploration"])

	if page == "Homepage":
		st.header("This is your data explorer.")
		st.write("Please select a page on the left.")
		st.subheader("Showing raw data....")	
		st.write(df.head())
	elif page == "Exploration":
		st.title("Data Exploration")
		x_axis = st.selectbox("Choose a variable for the x-axis", df.columns)
		y_axis = st.selectbox("Choose a variable for the y-axis", df.columns)
		visualize_data(df, x_axis, y_axis)

	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Recursive Feature Elimination"])

	if(choose_model == "Recursive Feature Elimination"):
			st.subheader("Which Feature is Important?")
			X_train, X_test, y_train, y_test = split(X,y)
			models = get_models()
			# evaluate the models and store results
			results, names = list(), list()
			for name, model in models.items():
				scores = evaluate_model(model,X_train,y_train)
				results.append(scores)
				names.append(name)
			model.fit(X_train, y_train)
			ypred = model.predict(X_test)
			predict_score = metrics.accuracy_score(y_test, ypred) * 100
			predict_report = classification_report(y_test, ypred)
			cm = metrics.confusion_matrix(y_test,ypred)
			st.subheader("Feature Selection Score")
			st.text("Accuracy of RFE Selection using DecisionTreeClassifier: ")
			st.write(round(mean(scores),2)*100,"%")
			pyplot.boxplot(results, labels=names, showmeans=True)
			st.pyplot()			
			st.subheader("We can also use the RFE model pipeline as a final model and make predictions for classification.")
			st.text("Accuracy of Prediction RFE using DecisionTreeClassifier is: ")
			st.write(predict_score,"%")
			st.text("Report of RFE using using DecisionTreeClassifier is: ")
			st.write(predict_report)
			sn.heatmap(cm, annot=True, annot_kws={"size": 16},cbar=False) # font size
			st.pyplot()

def test():
	df,X,y = loadData()
	X_train, X_test, y_train, y_test = split(X,y)
	models = get_models()
	# # evaluate the models and store results
	results, names = list(), list()
	for name, model in models.items():
		scores = evaluate_model(model,X_train,y_train)
		results.append(scores)
		names.append(name)
	model.fit(X_train, y_train)
	ypred = model.predict(X_test)
	predict_score = metrics.accuracy_score(y_test, ypred) * 100
	predict_report = classification_report(y_test, ypred)
	cm = metrics.confusion_matrix(y_test,ypred)
	print(round(mean(scores),2)*100,"%")
	pyplot.boxplot(results, labels=names, showmeans=True)
	print(predict_score,"%")
	st.text("Report of RFE using using DecisionTreeClassifier is: ")
	print(predict_report)
	sn.heatmap(cm, annot=True, annot_kws={"size": 16},cbar=False) # font size

if __name__ == "__main__":
	main()
	# test()