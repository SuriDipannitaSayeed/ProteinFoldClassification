# ProteinFoldClassificationwithGNNandPTGL
## Datasets
The Benchmark dataset folder contains the graphdata set generated from three different benchmark dataset. 1) DD dataset 2) EDD dataset 3) TG dataset. The Structural_Classification_Dataset contains the graph dataset for 4 different protein classes from SCOP and CATH. Those are: 1) All alpha graph 2) All beta graph 3) All alpha plus beta graph 4) All alpha by beta graph.
The following method have been used to convert into the pytorch graph from the PTGL gml file.
```sh
#Computing subgraph
#Alpha/Beta/Alpha-Beta Graph
import json
import torch
import torch
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
import networkx as nx
from pathlib import Path
def gml_toData_subgraph(attr_y,filename2,rangestr1,rangestr2):
  my_file2 = Path(filename2)
  data=0
  
  print(my_file2)
  if my_file2.is_file():
    try:
      G=nx.read_gml(filename2)
      #G.nodes(data="num_residues")
      l1=list(G.nodes())
      l=[]
      E1=list(G.edges())
      E=[]
      ids=[]
      #print(len(E1))
      if((rangestr1[0]=='-')and(rangestr1[:2]!='1G')):
        startend=rangestr1[1:].split("-")
        start=int(startend[0])
        end=int(startend[1])
      # if((rangestr1[0]=='-')and('1G' not in rangestr1)):
      #   startend=rangestr1[1:].split("-")
      #   start=int(startend[0])
      #   end=int(startend[1])
      if((rangestr1[0]!='-')and('317B' not in rangestr1)and('213A'not in rangestr1)and('223B' not in rangestr1)and('116A' not in rangestr1)and('7A' not in rangestr1)and('1C' not in rangestr1)and('1A' not in rangestr1 )and( '4A' not in rangestr1 )and('4P'not in rangestr1 )and('95C'not in rangestr1 )and('61P' not in rangestr1)and('1B' not in rangestr1)and('1G' not in rangestr1) and  ('1S' not in rangestr1) and('2A' not in rangestr1)and ('0A' not in rangestr1)and('1106A' not in rangestr1)and('106A' not in rangestr1)and('106B' not in rangestr1 )and('107A' not in rangestr1)and ('107B' not in rangestr1)):
        print(rangestr1)
        startend=rangestr1.split("-")
        start=int(startend[0])
        end=int(startend[1])
        # print(start)
        # print(end)
        start1=0
        end1=0

        if(len(rangestr2)>0):
          startend=rangestr2.split("-")
          start1=int(startend[0])
          end1=int(startend[1])
        print(G.edges())
        # print(G.nodes["0-H"])
        # print(G.nodes["0-H"]['pdb_res_start'])
        # print(G.nodes["0-H"]['pdb_res_end'])
        for i in range(len(l1)):
          st=G.nodes[l1[i]]['pdb_res_start']
          st=st[2:-2]
          #print(st)
          ed=G.nodes[l1[i]]['pdb_res_end']
          ed=ed[2:-2]
          #print(ed)
          if((int(st)>=start) and(int(ed)<=end)):
            l.append(l1[i])
            ids.append(i)

          
            #print(ed)
          if((int(st)>=start1) and(int(ed)<=end1)):
            l.append(l1[i])
            ids.append(i)

        print(l)
        for i in range(len(E1)):
          if((E1[i][0] in l)and (E1[i][1] in l)):
            E.append(E1[i])
        print(E)
        if((len(E)>0)and(len(l)>0)):
          #print(G.nodes[l[1]]['num_residues'])
          edge_atr=[]
          #countdegrees

          degrees=[]
          count=0
          
          for i in range(len(l)):
            for j in range(len(E)):
              if((E[j][0][:2]==l[i][:2])or (E[j][1][:2]==l[i][:2])):
                count=count+1
            
            degrees.append(count)
            count=0
          #print(degrees)
          #count_edges
          keys=[]
          values=[]
          edgetuple=[]
          #print(E)
          for j in range(len(E)):
            #print(E[j][0][:-2])
            #keys.append(int(E[j][0][:-2]))
            #values.append(int(E[j][1][:-2]))
            #if(E[j][0] in l):
            a1=(l.index(E[j][0]),l.index(E[j][1]))
            edgetuple.append(a1)
            a1=(l.index(E[j][1]),l.index(E[j][0]))
            edgetuple.append(a1)
            #keys.append(int(E[j][1][:-2]))
            #values.append(int(E[j][0][:-2]))

          edgetuple=sorted(list(edgetuple))
          for i in range(len(edgetuple)):
              keys.append(edgetuple[i][0])
              values.append(edgetuple[i][1])
          #node_type
          node_type=[]
          for i in range(len(l)):
          
            if(l[i][len(l[i])-1]=='H'):
              node_type.append(1)
            if(l[i][len(l[i])-1]=='E'):
              node_type.append(2)

          #print(node_type)

          #SSE_length
          num_residues=[]
          for i in range(len(l)):
            num_residues.append(int(G.nodes[l[i]]['num_residues']))

          #print(num_residues)

          edge_index = torch.tensor([(keys),(values)], dtype=torch.long)
          edge_attr = torch.tensor([edge_atr]).view(-1, 1).long()
          y=torch.tensor([attr_y], dtype=torch.long)
          x=torch.cat([torch.tensor(degrees).view(-1, 1).float(),torch.tensor(node_type).view(-1, 1).float(),torch.tensor(num_residues).view(-1, 1).float()],-1)
          data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=y) 

    except nx.NetworkXError:
      print("not found")
  return data

#full-graph
#Alpha/Beta/Alpha-Beta Graph
import json
import torch
import torch
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
import networkx as nx
from pathlib import Path
def gml_toData_graph(attr_y,filename2):
  my_file2 = Path(filename2)
  data=0
  
  print(my_file2)
  if my_file2.is_file():
    try:
      G=nx.read_gml(filename2)
      #G.nodes(data="num_residues")
      l=list(G.nodes())
      E=list(G.edges())
      print(len(E))
      # startend=rangestr.split("-")
   
      # print(int(startend[0]))
      # print(int(startend[1]))
      print(G.edges())
      # print(G.nodes["0-H"])
      # print(G.nodes["0-H"]['pdb_res_start'])
      # print(G.nodes["0-H"]['pdb_res_end'])
      if((len(E)>0)and(len(l)>0)):
        #print(G.nodes[l[1]]['num_residues'])
        edge_atr=[]
        #countdegrees

        degrees=[]
        count=0
        
        for i in range(len(l)):
          for j in range(len(E)):
            if((E[j][0][:2]==l[i][:2])or (E[j][1][:2]==l[i][:2])):
              count=count+1
          
          degrees.append(count)
          count=0
        #print(degrees)
        #count_edges
        keys=[]
        values=[]
        edgetuple=[]
        #print(E)
        for j in range(len(E)):
          #print(E[j][0][:-2])
          #keys.append(int(E[j][0][:-2]))
          #values.append(int(E[j][1][:-2]))
          a1=(int(E[j][0][:-2]),int(E[j][1][:-2]))
          edgetuple.append(a1)
          a1=(int(E[j][1][:-2]),int(E[j][0][:-2]))
          edgetuple.append(a1)
          #keys.append(int(E[j][1][:-2]))
          #values.append(int(E[j][0][:-2]))

        edgetuple=sorted(list(edgetuple))
        for i in range(len(edgetuple)):
            keys.append(edgetuple[i][0])
            values.append(edgetuple[i][1])
        #node_type
        node_type=[]
        for i in range(len(l)):
        
          if(l[i][len(l[i])-1]=='H'):
            node_type.append(1)
          if(l[i][len(l[i])-1]=='E'):
            node_type.append(2)

        #print(node_type)

        #SSE_length
        num_residues=[]
        for i in range(len(l)):
          num_residues.append(int(G.nodes[l[i]]['num_residues']))

        #print(num_residues)

        edge_index = torch.tensor([(keys),(values)], dtype=torch.long)
        edge_attr = torch.tensor([edge_atr]).view(-1, 1).long()
        y=torch.tensor([attr_y], dtype=torch.long)
        x=torch.cat([torch.tensor(degrees).view(-1, 1).float(),torch.tensor(node_type).view(-1, 1).float(),torch.tensor(num_residues).view(-1, 1).float()],-1)
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=y) 

    except nx.NetworkXError:
      print("not found")
  return data

### Parameter Settings

|Datasets|	lr|	weight_decay|	batch_size|	pool_ratio	|dropout|	net_layers|



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
# Google Colab

[https://colab.research.google.com/drive/1oXu43mqr3H1TLzSIF_6IQ9xjSlA7voNg#scrollTo=V3cW_VgVJmgg]
