# Neural-Networks
A multi-layer neural networks for ten-class classification - model fit using progagation & backpropagation

## Predicted labels
The **predicted_labels.h5** will be generated in the **code/output/** folder after running the main program to predict the labels
* predicted_labels.h5

## Setup environment and prepare the input data
We suggest you use **python 3.6+** as the runtime

### Use virtualenv
We use python virtualenv to separate the dependencies with other projects, to run and activate it please run the following
```
pip install virtualenv
```
Create a virtual env
```
virtualenv comp5329-assignment1
```
To activate it
```
source comp5329-assignment1/bin/activate
```

### Install dependencies for the program
**please note that scipy==1.2.1 is a must**
```
pip install -r requirements.txt
```

### Prepare input data
Please copy the required input data into the *code/input/* folder, make sure the following files are in place before running the programs
* train_128.h5
* train_label.h5
* test_128.h5

## Running the main program
to predict labels for test data provided in args test_data
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json --test_data=test_128.h5
```

or to predict labels as well as display the test accuracy comparing with the true test label file provided in args test_labels
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json --test_data=test_128.h5 --test_labels=test_label.h5
```

or to perform model training only
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json
```

## Zip for submission
```
zip -r 480458339_470325230.zip submission/ -x submission/code/input/* -x submission/code/algorithm/bak/*
```