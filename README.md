# Neural Architecture Search with Genetic Algorithms

good architectures can make more robust and easy to train models.
Searching for a good architecture usually requires tons of comutations time because the most common way to evaluate an architecture is: train it and then assign it a score based on the accuracy (or some other metric).

To speed up the reasearch we do not train our models (as proposed by [others] too). We instead use some heuristics to predict how good a network is.
