Theano tutorial
==========================

This repository contains contains code examples for the Theano tutorial at [http://www.marekrei.com/blog/theano-tutorial/](http://www.marekrei.com/blog/theano-tutorial/)

Minimal Working Example
--------------------------
Basically the smallest Theano example I could come up with. 
It calculates the dot product between vectors [0.2, 0.9] and [1.0, 1.0].

Run with:

	python minimal_working_example.py


The script should print value 0.9



Minimal Training Example
-------------------------

The script iteratively modifies the first vector in the previous example, using gradient descent, such that the dot product would have value 20.

Run with:

	python minimal_training_example.py

It should print the value at each of the 10 iterations:

	0.9
	8.54
	13.124
	15.8744
	17.52464
	18.514784
	19.1088704
	19.46532224
	19.679193344
	19.8075160064


Simple Classifier Example
---------------------------

The next example tries to train a small network on tiny (but real) dataset. 
The task is to predict whether the GDP per capita for a country is more than the average GDP, based on the following features:

* Population density (per suqare km)
* Population growth rate (%)
* Urban population (%)
* Life expectancy at birth (years)
* Fertility rate (births per woman)
* Infant mortality (deaths per 1000 births)
* Enrolment in tertiary education (%)
* Unemployment (%)
* Estimated control of corruption (score)
* Estimated government effectiveness (score)
* Internet users (per 100 people)

The *data/* directory contains the files for training (121 countries) and testing (40 countries). 
Each row represents one country, the first column is the label, followed by the features.
The feature values have been normalised, by subtracting the mean and dividing by the standard deviation. 
The label is 1 if the GDP is more than average, and 0 otherwise.


Run with:

	python classifier.py data/countries-classify-gdp-normalised.train.txt data/countries-classify-gdp-normalised.test.txt

The script will print information about 10 training epochs and the result on the test set:

	Epoch: 0, Training_cost: 28.4304042768, Training_accuracy: 0.578512396694
	Epoch: 1, Training_cost: 24.5186290354, Training_accuracy: 0.619834710744
	Epoch: 2, Training_cost: 22.1283727037, Training_accuracy: 0.619834710744
	Epoch: 3, Training_cost: 20.7941253329, Training_accuracy: 0.619834710744
	Epoch: 4, Training_cost: 19.9641569475, Training_accuracy: 0.619834710744
	Epoch: 5, Training_cost: 19.3749411377, Training_accuracy: 0.619834710744
	Epoch: 6, Training_cost: 18.8899216914, Training_accuracy: 0.619834710744
	Epoch: 7, Training_cost: 18.4006371608, Training_accuracy: 0.677685950413
	Epoch: 8, Training_cost: 17.7210185975, Training_accuracy: 0.793388429752
	Epoch: 9, Training_cost: 16.315597037, Training_accuracy: 0.876033057851
	Test_cost: 5.01800578051, Test_accuracy: 0.925




RNN Classifier Example
-------------------------

Now let's try a real task of realistic size - sentiment classification on the Stanford sentiment corpus.
We will use a recurrent neural network for this, to show how they work in Theano.

The task is to classify sentences into 5 classes, based on their fine-grained sentiment (very negative, slightly negative, neutral, slightly positive, very positive).
We use the dataset published in "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank" (Socher et al., 2013).

Start by downloading the dataset from [http://nlp.stanford.edu/sentiment/](http://nlp.stanford.edu/sentiment/) (the main zip file) and unpack it somewhere.
Then, create training and test splits in the format that is more suitable for us, using the provided script in this repository:

	python stanford_sentiment_extractor.py 1 full /path/to/sentiment/dataset/ > data/sentiment.train.txt
	python stanford_sentiment_extractor.py 2 full /path/to/sentiment/dataset/ > data/sentiment.test.txt

Now we can run the classifier with:

	python rnnclassifier.py data/sentiment.train.txt data/sentiment.test.txt

The code will train for 3 passes over the training data, and will then print performance on the test data. 

	Epoch: 0	Cost: 25929.9481023	Accuracy: 0.286633895131
	Epoch: 1	Cost: 21541.7328736	Accuracy: 0.35779494382
	Epoch: 2	Cost: 17857.7320117	Accuracy: 0.443586142322
	Test_cost: 4934.24376649	Test_accuracy: 0.349773755656

The accuracy on the test set is about 38%, which isn't a great result. But it is quite a difficult task - the current state-of-the-art system ([Tai ei al., 2015](https://aclweb.org/anthology/P/P15/P15-1150.pdf)) achieves 50.9% accuracy, using a large amount of additional phrase-level annotations, and a much bigger network based on LSTMs and parse trees. As there are 5 classes to choose from, a random system would get 20% accuracy.






