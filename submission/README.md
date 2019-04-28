# Neural-Networks
A multi-layer neural networks for multi-class classification
progagation & backpropagation

## Running the Program
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
```
pip install -r requirements.txt
```

### Run the main program

to fit the model and perform training only
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json
```
or to perform cross validation for the model
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json --cv_data=cv_128.h5 --cv_labels=cv_label.h5
```
or to generate predicted labels by providing test_data in args
```
python multilayer_neutral_network.py --config=8987.json --test_data=test_128.h5
```

## Zip for submission
```
zip -r 480458339_470325230.zip submission/ -x submission/code/input/*
```