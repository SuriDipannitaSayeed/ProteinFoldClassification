# ProteinFoldClassification
# Google Colab

[https://colab.research.google.com/drive/1oXu43mqr3H1TLzSIF_6IQ9xjSlA7voNg#scrollTo=V3cW_VgVJmgg]
## Datasets
The Benchmark dataset folder contains the graphdata set generated from five different benchmark dataset. 1) LINDAHL dataset 2) SCOP_TEST dataset 3) DD dataset 4) RDD dataset 5) EDD dataset. The Structural_Classification_Dataset contains the graph dataset for 5 different protein classes. 1) All alpha graph 2) All beta graph 3) All alpha plus beta graph 4) All alpha by beta graph 5) All alpha plus beta multidomain graph.
### Parameter Settings

|Datasets|	lr|	weight_decay|	batch_size|	pool_ratio	|dropout|	net_layers|

|LINDAHL|0.005|0.001|512|0.5|0.1|3|

|SCOP_TEST|0.005|0.001|256|0.5|0.1|3|

|DD|0.005|0.001|512|0.5|0.1|3|

|RDD|0.005|0.001|512|0.5|0.1|3|

|EDD|0.005|0.001|512|0.5|0.1|3|
#### Run
To run simply execute this cell in google collaboratory environment.
Run:

```sh
!python /content/main.py
```
or execute the main programme cell uploading all the necessary files. 1) layers.py 2) models.py 3)sparse_softmax.py  and the intended dataset.
The dataset name in the main program should have to be changed accordingly.

    
```sh
with open("/content/all_betagraph_.txt","rb") as f:
    traindataset = pickle.load(f,encoding="latin1")
with open("/content/all_betagraph_.txt","rb") as f:
    testdataset = pickle.load(f,encoding="latin1")
```
