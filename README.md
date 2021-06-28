# ProteinFoldClassificationwithGNN
# Google Colab

[https://colab.research.google.com/drive/1oXu43mqr3H1TLzSIF_6IQ9xjSlA7voNg#scrollTo=V3cW_VgVJmgg]
## Datasets
The Benchmark dataset folder contains the graphdata set generated from five different benchmark dataset. 1) LINDAHL dataset 2) SCOP_TEST dataset 3) DD dataset 4) RDD dataset 5) EDD dataset. The Structural_Classification_Dataset contains the graph dataset for 5 different protein classes. 1) All alpha graph 2) All beta graph 3) All alpha plus beta graph 4) All alpha by beta graph 5) All alpha plus beta multidomain graph.
The following method have been used to convert into the pytorch graph from the PTGL json file.
```sh
import json
import torch
import torch
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
def json_toData_(attr_y,filename):
  Dataset=[]
  with open(filename) as f:
    data = json.load(f)
  #print(list(data.keys()))
  vertices=data['vertices']
  edges=data['edges']
  edge_labels=data['edgeLabels']
  #print(vertices)
  edges_keys=list(edges.keys())
  edges_values=list(edges.values())
  ek=list(edge_labels.keys())
  ev=list(edge_labels.values())
  keys=[]
  values=[]
  degrees=[]
  vertex_labels=[]
  edge_atr=[]
  edge_tuple=[]
  #print(len(edges_values))
  for j in range(len(edges_values)):
      for k in range(len(edges_values[j])):
        keys.append((j))
  for i in range(len(edges_values)):
    degrees.append(len(edges_values[i]))
    if((vertices[i][0])=='h'):
      vertex_labels.append(1)
    if((vertices[i][0])=='e'):
      vertex_labels.append(2)
    
    for j in range(len(edges_values[i])):
      #keys.append(int(edges_keys[i][1]))
      a=((edges_keys[i][1]))
      #a1=(int(edges_keys[i][1]),int(edges_values[i][j][1]))
      #edge_tuple.append(a1)
      values.append(int(edges_values[i][j][1]))
      b=(edges_values[i][j][1])
      for l in range(len(ek)):
        if(((ek[l][1]==a)and(ek[l][4]==b))or((ek[l][1]==b)and(ek[l][4]==a))):
          if(ev[l]=='m'):
            edge_atr.append(1)
          if(ev[l]=='a'):
            edge_atr.append(2)
          if(ev[l]=='p'):
            edge_atr.append(3)

  #print(edges) 

  #print(edgetuple)
  for i in range(len(keys)):
    if(keys[i]!=values[i]):
      a1=(keys[i],values[i])
      edge_tuple.append(a1)
  #print(edge_tuple)
  edgetuple=set(edge_tuple)
  #print(edgetuple)
  edgetuple=sorted(list(edgetuple))
  #print(edgetuple)
  keys=[]
  values=[]
  for i in range(len(edgetuple)):
    keys.append(edgetuple[i][0])
    values.append(edgetuple[i][1])

  edge_index = torch.tensor([(keys),(values)], dtype=torch.long)
  edge_attr = torch.tensor([edge_atr]).view(-1, 1).long()
 
  y=torch.tensor([attr_y], dtype=torch.long)
  #print(degrees) 
  #print(vertex_labels) 
  x=torch.cat([torch.tensor(degrees).view(-1, 1).float(),torch.tensor(vertex_labels).view(-1, 1).float()],-1)
  #print(ek[0][4])
  #print(ev)
  #print(edge_labels.keys()) 
  #print(len(ek)) 
  #print(edge_labels)
  #print(edge_attr)

  data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=y)
  #print(data)
  #G = to_networkx(data, node_attrs=['x'])
  #G=G.to_undirected()
  #g = dgl.from_networkx(G, node_attrs=['x'])
  
  #data=data.coalesce()
  
  #print(data)
  
  #print(data)
  return data
```
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
