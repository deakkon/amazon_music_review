# Amazon Music Revew as classification - solution details

This is a PoC solution, targeting Amazon Music Dataset review rating classification. 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the needed requirements. It is highly suggested to create a virtual environment for this PoC! This can be done either with virtenv or conda. The one on my machine was created

```bash
conda create --name amr python=3.8
conda activate amr
```

Once created and activated, install the needed requirements:
```bash
pip install -r requirements.txt
```

## Usage

The code in this repo has two intentions:
- predict using a fine-tuned BERTlike model, though a web service/endpoint.
- train a new model using a new dataset, using a BERTlike model.

First, you need to get the code. For the purposes of this assigment, the code is avaliable in XXX. 

```console
cd /path/to/repo
export PYTHONPATH=$(pwd)
```

All commands are executed from the projects root. 

### Predict with supplied model

To predict on new datapoints, you need to start the server. Tornado was used for this PoC. To start the server execute

```console
python src/server/server.py
```


The server will start on 
```console
http://localhost:5555/
```

and excepts POST requests. To simulate a request, Postman was used while developing. To test a dummy round trip, you can also use curl:

```console
curl -L -X POST "http://localhost:5555" -H 'Content-Type: application/json' --data-raw '{"summary": "This little piggy went to the bank.","review": "And this one to the mall. He was a spender."}'
```

This will return a JSON object containing the predicted target label for the input datapoint:

```python
{"someList": [5]}
```

Notice the JSON key names; they need to be exactly the same in any other request.

### Train a new model

Training a new model is also straitghforward. One needs to: 
1. (optionally) Prepare a dataset consistin of 3 columns: first two columns are strings, following the same pattern as in the train file. The last column is the target column. Pass this using the -fn parameter. If no file is defined, the train data supplied with the assigment will be used; 
2. (optionally) Define a path where to serialize the model. If no path is given, the model will be saved to BERT_ARTIFACTS specified in definitions.py. It is suggested to supply this path when training on external data to avoid overwriting the model supplied with this solution!

Training is executed as follows:

```console
python src/model/bertlike/trainer.py -fn path/to/data -sn path/to/model/serialization/folder
```

where data/training_new.csv is a dummy training dataset. 

One minor issue at this point is the lack of model versioning and intelligent serialization of trained model and other artifacts (i.e. due to time constraints for developing this solution, using non-default paths has not been thoroughly tested!). The solution also lacks any batch processing of requests. 
