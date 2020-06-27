
# Network Science Final Project
Roberto & Gacera

## Data Wrangling
### Conversion to nodes and edges of the 2019 election city-level results in Region I (Ilocos)
 - Network: bipartite (directed)
 - Nodes: cities and politicians
 - Edges: exists if a city voted for a politician (within a given threshold)
 - Weight: normalized votes (votes/votes of the winning candidate)


```python
import pandas as pd
import pickle
import re

with open('city_votes_2019.pkl', 'rb') as f:
    df = pickle.load(f)
```

### Filter to Ilocos region


```python
df = df[df.region=='REGION I']
```

#### Retrieve party from candidate official election name


```python
df['party'] = df.candidate.apply(lambda x: re.findall(r'\((.*)\)', x)[0] if len(re.findall(r'\(.*\)', x)) > 0 else '')
df.loc[df.party=='', 'party'] = 'PARTY LIST'
```


```python
df.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>region</th>
      <th>province</th>
      <th>city</th>
      <th>position</th>
      <th>candidate</th>
      <th>candidate_party</th>
      <th>votes</th>
      <th>total_votes</th>
      <th>percentage</th>
      <th>party</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>REGION I</td>
      <td>ILOCOS NORTE</td>
      <td>ADAMS</td>
      <td>SENATOR</td>
      <td>ABEJO, VANGIE (IND)</td>
      <td>LGBTQ PARTY</td>
      <td>6</td>
      <td>5366</td>
      <td>0.0011</td>
      <td>IND</td>
    </tr>
    <tr>
      <th>1</th>
      <td>REGION I</td>
      <td>ILOCOS NORTE</td>
      <td>BACARRA</td>
      <td>SENATOR</td>
      <td>ABEJO, VANGIE (IND)</td>
      <td>LGBTQ PARTY</td>
      <td>89</td>
      <td>87217</td>
      <td>0.0010</td>
      <td>IND</td>
    </tr>
  </tbody>
</table>
</div>



#### Determine winning candidates by getting the maximum percentage value per position per city


```python
maxs = df.groupby(['region', 'province', 'city', 'position']).percentage.max().reset_index()
maxs.columns = ['region', 'province', 'city', 'position', 'maxs']
```


```python
df = pd.merge(df, maxs)
```

#### Compute for weight of the edges: votes/ votes of winning candidate


```python
df['per_sc'] = df.percentage/df.maxs
```


```python
df.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>region</th>
      <th>province</th>
      <th>city</th>
      <th>position</th>
      <th>candidate</th>
      <th>candidate_party</th>
      <th>votes</th>
      <th>total_votes</th>
      <th>percentage</th>
      <th>party</th>
      <th>maxs</th>
      <th>per_sc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>REGION I</td>
      <td>ILOCOS NORTE</td>
      <td>ADAMS</td>
      <td>SENATOR</td>
      <td>ABEJO, VANGIE (IND)</td>
      <td>LGBTQ PARTY</td>
      <td>6</td>
      <td>5366</td>
      <td>0.0011</td>
      <td>IND</td>
      <td>0.1731</td>
      <td>0.006355</td>
    </tr>
    <tr>
      <th>1</th>
      <td>REGION I</td>
      <td>ILOCOS NORTE</td>
      <td>ADAMS</td>
      <td>SENATOR</td>
      <td>AFUANG, ABNER (WPP)</td>
      <td>LABOR PARTY PHILIPPINES</td>
      <td>5</td>
      <td>5366</td>
      <td>0.0009</td>
      <td>WPP</td>
      <td>0.1731</td>
      <td>0.005199</td>
    </tr>
  </tbody>
</table>
</div>



#### Filter to the top N candidates per position
 - senator: top 30
 - house of representative: top 5
 - mayor: top 5
 - vice mayor: top 5
 - party list: top 51
 - governor: all since there were only a few candidates
 - vice governor: all since there were only a few candidates


```python
sorted(df.position.unique())
```




    ['MAYOR',
     'MEMBER, HOUSE OF REPRESENTATIVES',
     'PARTY LIST',
     'PROVINCIAL GOVERNOR',
     'PROVINCIAL VICE-GOVERNOR',
     'SENATOR',
     'VICE-MAYOR']




```python
df = df.sort_values(['region', 'province', 'city', 'position', 'percentage'], ascending=False)
sen = df[df.position=='SENATOR']
rep = df[df.position=='MEMBER, HOUSE OF REPRESENTATIVES']
mayor = df[df.position=='MAYOR']
vmayor = df[df.position=='VICE-MAYOR']
plist = df[df.position=='PARTY LIST']
gov = df[df.position=='PROVINCIAL GOVERNOR']
vgov = df[df.position=='PROVINCIAL VICE-GOVERNOR']
```


```python
sens = sen.groupby(['region', 'province', 'city', 'position']).head(30)
reps = rep.groupby(['region', 'province', 'city', 'position']).head(5)
mayors = mayor.groupby(['region', 'province', 'city', 'position']).head(5)
vmayors = vmayor.groupby(['region', 'province', 'city', 'position']).head(5)
plists = plist.groupby(['region', 'province', 'city', 'position']).head(51)
```


```python
df = pd.concat([sens, reps, mayors, vmayors, plists, gov, vgov]).drop(columns=['maxs'])
```


```python
df.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>region</th>
      <th>province</th>
      <th>city</th>
      <th>position</th>
      <th>candidate</th>
      <th>candidate_party</th>
      <th>votes</th>
      <th>total_votes</th>
      <th>percentage</th>
      <th>party</th>
      <th>per_sc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7687</th>
      <td>REGION I</td>
      <td>PANGASINAN</td>
      <td>VILLASIS</td>
      <td>SENATOR</td>
      <td>VILLAR, CYNTHIA (NP)</td>
      <td>NACIONALISTA PARTY</td>
      <td>21830</td>
      <td>277610</td>
      <td>0.0786</td>
      <td>NP</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7680</th>
      <td>REGION I</td>
      <td>PANGASINAN</td>
      <td>VILLASIS</td>
      <td>SENATOR</td>
      <td>POE, GRACE (IND)</td>
      <td>LGBTQ PARTY</td>
      <td>21041</td>
      <td>277610</td>
      <td>0.0757</td>
      <td>IND</td>
      <td>0.963104</td>
    </tr>
  </tbody>
</table>
</div>



### Create nodes table

#### Retrieve unique cities
Since some cities have the same name, get the unique cities using `province` and `city` columns.


```python
cities = df[['province', 'city']].drop_duplicates().reset_index(drop=True).reset_index()
```

#### Set node type to CITY


```python
cities['Node_Type'] = 'CITY'
```


```python
cities.columns = ['Id', 'province', 'city', 'Node_Type']
```


```python
cities['Label'] = cities.province + '/' + cities.city
```


```python
cities = cities[['Id', 'Label', 'Node_Type']]
```


```python
cities.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Label</th>
      <th>Node_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>120</th>
      <td>120</td>
      <td>ILOCOS NORTE/BANNA (ESPIRITU)</td>
      <td>CITY</td>
    </tr>
    <tr>
      <th>121</th>
      <td>121</td>
      <td>ILOCOS NORTE/BANGUI</td>
      <td>CITY</td>
    </tr>
    <tr>
      <th>122</th>
      <td>122</td>
      <td>ILOCOS NORTE/BADOC</td>
      <td>CITY</td>
    </tr>
    <tr>
      <th>123</th>
      <td>123</td>
      <td>ILOCOS NORTE/BACARRA</td>
      <td>CITY</td>
    </tr>
    <tr>
      <th>124</th>
      <td>124</td>
      <td>ILOCOS NORTE/ADAMS</td>
      <td>CITY</td>
    </tr>
  </tbody>
</table>
</div>



#### Retrieve unique politicians


```python
pols = df[['candidate', 'party']].drop_duplicates().reset_index(drop=True).reset_index()
```


```python
pols.columns = ['Id', 'Label', 'Node_Type']
```

#### Change Id such that it continues from the city Id values


```python
pols.Id = pols.Id + 125
```


```python
pols.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Label</th>
      <th>Node_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>733</th>
      <td>858</td>
      <td>AQUINO, JULIUS (KDP)</td>
      <td>KDP</td>
    </tr>
    <tr>
      <th>734</th>
      <td>859</td>
      <td>SINGSON, JERRY (BILEG)</td>
      <td>BILEG</td>
    </tr>
    <tr>
      <th>735</th>
      <td>860</td>
      <td>ZARAGOZA, ANICKA (PDPLBN)</td>
      <td>PDPLBN</td>
    </tr>
    <tr>
      <th>736</th>
      <td>861</td>
      <td>MARCOS, MARIANO II (NP)</td>
      <td>NP</td>
    </tr>
    <tr>
      <th>737</th>
      <td>862</td>
      <td>RAMONES, MICHAEL (PDPLBN)</td>
      <td>PDPLBN</td>
    </tr>
  </tbody>
</table>
</div>



#### Combine cities and politicians into 1 table


```python
nodes = pd.concat([cities, pols]).reset_index(drop=True)
```


```python
with open('ILOCOS_nodes_2019.pkl', 'wb') as f:
    pickle.dump(nodes, f)
```


```python
nodes.to_csv('ILOCOS_nodes_2019.csv', index=False)
```

### Filter rows with only the significant normalized votes (>= median)


```python
df.per_sc.describe()
```




    count    11480.000000
    mean         0.221897
    std          0.309492
    min          0.000000
    25%          0.014144
    50%          0.060663
    75%          0.316539
    max          1.000000
    Name: per_sc, dtype: float64




```python
df.shape
```




    (11480, 11)




```python
df = df[df.per_sc>=0.060663]
```


```python
df.shape
```




    (5740, 11)




```python
df['prov_city'] = df.province + '/' + df.city
```

### Create combined table of main dataframe and node labels (for future reference)


```python
df = pd.merge(df, nodes, left_on='prov_city', right_on='Label')
```


```python
df = pd.merge(df, nodes, left_on='candidate', right_on='Label')
```


```python
df.shape
```




    (5740, 18)




```python
df.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>region</th>
      <th>province</th>
      <th>city</th>
      <th>position</th>
      <th>candidate</th>
      <th>candidate_party</th>
      <th>votes</th>
      <th>total_votes</th>
      <th>percentage</th>
      <th>party</th>
      <th>per_sc</th>
      <th>prov_city</th>
      <th>Id_x</th>
      <th>Label_x</th>
      <th>Node_Type_x</th>
      <th>Id_y</th>
      <th>Label_y</th>
      <th>Node_Type_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>REGION I</td>
      <td>PANGASINAN</td>
      <td>VILLASIS</td>
      <td>SENATOR</td>
      <td>VILLAR, CYNTHIA (NP)</td>
      <td>NACIONALISTA PARTY</td>
      <td>21830</td>
      <td>277610</td>
      <td>0.0786</td>
      <td>NP</td>
      <td>1.000000</td>
      <td>PANGASINAN/VILLASIS</td>
      <td>0</td>
      <td>PANGASINAN/VILLASIS</td>
      <td>CITY</td>
      <td>125</td>
      <td>VILLAR, CYNTHIA (NP)</td>
      <td>NP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>REGION I</td>
      <td>PANGASINAN</td>
      <td>URDANETA CITY</td>
      <td>SENATOR</td>
      <td>VILLAR, CYNTHIA (NP)</td>
      <td>NACIONALISTA PARTY</td>
      <td>42312</td>
      <td>586456</td>
      <td>0.0721</td>
      <td>NP</td>
      <td>0.990385</td>
      <td>PANGASINAN/URDANETA CITY</td>
      <td>1</td>
      <td>PANGASINAN/URDANETA CITY</td>
      <td>CITY</td>
      <td>125</td>
      <td>VILLAR, CYNTHIA (NP)</td>
      <td>NP</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open('ILOCOS_df_2019.pkl', 'wb') as f:
    pickle.dump(df, f)
```

### Create edges table


```python
edges = df[['Id_x', 'Id_y', 'per_sc', 'position']]
```


```python
edges.columns = ['Source', 'Target', 'Weight', 'Position']
```

#### Edge table:
- Source: city Id
- Target: politician Id
- Weight: normalized vote


```python
edges.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Weight</th>
      <th>Position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>125</td>
      <td>1.000000</td>
      <td>SENATOR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>125</td>
      <td>0.990385</td>
      <td>SENATOR</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open('ILOCOS_edges_2019.pkl', 'wb') as f:
    pickle.dump(edges, f)
```


```python
edges.to_csv('ILOCOS_edges_2019.csv', index=False)
```


```python

```
