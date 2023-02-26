# Objectives

The learning objectives of this assignment are to:
1. learn the TensorFlow Keras APIs for convolutional and recurrent neural
   networks.
2. explore the space of hyper-parameters for convolutional and recurrent
   networks.

# Environment Setup

* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [tensorflow (version 2.9)](https://www.tensorflow.org/)
* [pytest](https://docs.pytest.org/)

# Write your code

In this project, we will implement several convolutional and recurrent neural networks using the
[TensorFlow Keras API](https://www.tensorflow.org/guide/keras/).

# Tests

Tests have been provided in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
...
============================== 4 failed in 6.04s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py
0.6 RMSE for RNN on toy problem
.
89.0% accuracy for CNN on MNIST sample
.
88.9% accuracy for RNN on Youtube comments
.
85.4% accuracy for CNN on Youtube comments
.                                                          [100%]

============================== 4 passed in 56.97s ==============================
```
**Warning**: The performance of your models may change somewhat from run to run,
especially when moving from one machine to another, since neural network models
are randomly initialized.

# Acknowledgments

The author of the test suite (test_nn.py) is Dr. Steven Bethard.
