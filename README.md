Theano tutorial (sample code)
==========================


Minimal Working Example
--------------------------

Run with:

	python minimal_working_example.py

It calculates the dot product between vectors [0.2, 0.9] and [1.0, 1.0].
The script should print value 0.9



Minimal Training Example
-------------------------

Run with:

	python minimal_training_example.py

The script iteratively modifies the first vector in the previous example, using gradient descent, such that the dot product would have value 20.
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
