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
### please note that scipy==1.2.1 is a must
```
pip install -r requirements.txt
```

### Run the main program

to perform training only
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json
```

or to predict labels for test data provided in args test_data
```
python multilayer_neutral_network.py --config=8987.json --test_data=test_128.h5
```

or to predict labels as well as output a test accuracy calculated from the true test label file
```
cd ./code/algorithm
python multilayer_neutral_network.py --config=8987.json --test_data=test_128.h5 --test_labels=test_label.h5
```
## Zip for submission
```
zip -r 480458339_470325230.zip submission/ -x submission/code/input/*
```