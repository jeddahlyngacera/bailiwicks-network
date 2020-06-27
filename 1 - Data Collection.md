
# Network Science Final Project
Roberto & Gacera

## Data Collection
2019 election results is collected from jojie `/mnt/data/public/elections/nle2019/contests/`


```python
import sqlite3
import pandas as pd
import glob
import re
import datetime
import pickle
import glob
```

### Get all 2019 election contest files


```bash
%%bash --out contests_file
find /mnt/data/public/elections/nle2019/contests/ -type f -name *.json
```


```python
contests = [i for i in contests_file.split('\n') if i]
```


```python
contests[0]
```




    '/mnt/data/public/elections/nle2019/contests/4438.json'




```python
values = []
for c in contests:
    with open(c, 'r') as f:
        data = json.load(f)
    for i in data['bos']:
        values.append((c, data['cc'], data['cn'], data['ccn'], i['boc'], i['bon'], i['pn']))
```


```python
values[:2]
```




    [('/mnt/data/public/elections/nle2019/contests/4438.json',
      4438,
      'MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGUI   - LONE DIST',
      'COUNCILOR',
      20963,
      'ACOBA, ROGELIO (NP)',
      'NACIONALISTA PARTY'),
     ('/mnt/data/public/elections/nle2019/contests/4438.json',
      4438,
      'MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGUI   - LONE DIST',
      'COUNCILOR',
      20964,
      'BALBAG, ROGERICK (NP)',
      'NACIONALISTA PARTY')]




```python
conn = sqlite3.connect('elvotes2019.db')
conn.executescript('''
DROP TABLE IF EXISTS prep1_2019;
CREATE TABLE prep1_2019 (
    c_file      VARCHAR,
    cc          VARCHAR,
    cn          VARCHAR,
    ccn         VARCHAR,
    bo          VARCHAR,
    bon         VARCHAR,
    pn          VARCHAR
);
''')
conn.commit()

conn.executemany('''INSERT INTO prep1_2019 VALUES (?, ?, ?, ?, ?, ?, ?)''', values)
conn.commit()
```


```python
df = pd.read_sql('SELECT * FROM prep1_2019', conn)
```


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c_file</th>
      <th>cc</th>
      <th>cn</th>
      <th>ccn</th>
      <th>bo</th>
      <th>bon</th>
      <th>pn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/mnt/data/public/elections/nle2019/contests/44...</td>
      <td>4438</td>
      <td>MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGU...</td>
      <td>COUNCILOR</td>
      <td>20963</td>
      <td>ACOBA, ROGELIO (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/mnt/data/public/elections/nle2019/contests/44...</td>
      <td>4438</td>
      <td>MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGU...</td>
      <td>COUNCILOR</td>
      <td>20964</td>
      <td>BALBAG, ROGERICK (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/mnt/data/public/elections/nle2019/contests/44...</td>
      <td>4438</td>
      <td>MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGU...</td>
      <td>COUNCILOR</td>
      <td>20965</td>
      <td>CAMPAÑANO, ANTHONY (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/mnt/data/public/elections/nle2019/contests/44...</td>
      <td>4438</td>
      <td>MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGU...</td>
      <td>COUNCILOR</td>
      <td>20966</td>
      <td>DOLDOLEA, NORMA (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/mnt/data/public/elections/nle2019/contests/44...</td>
      <td>4438</td>
      <td>MEMBER, SANGGUNIANG BAYAN ILOCOS NORTE - BANGU...</td>
      <td>COUNCILOR</td>
      <td>20968</td>
      <td>FAYLOGNA, SUSAN (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (43451, 7)



### Filter positions needed for analysis


```python
positions_needed = ['SENATOR', 'PARTY LIST', 'MEMBER, HOUSE OF REPRESENTATIVES',
                    'PROVINCIAL GOVERNOR', 'PROVINCIAL VICE-GOVERNOR',
                    'MAYOR', 'VICE-MAYOR']
```


```python
df = df[df.ccn.isin(positions_needed)]
```


```python
df.shape
```




    (8843, 7)



### Get all 2019 election result files


```python
coc_files = glob.glob('/mnt/data/public/elections/nle2019/results/*/*/*/coc.json')
```


```python
cocs = [[i] + i[43:].split('/')[:-1] for i in coc_files]
```


```python
cocs[0]
```




    ['/mnt/data/public/elections/nle2019/results/REGION I/ILOCOS NORTE/ADAMS/coc.json',
     'REGION I',
     'ILOCOS NORTE',
     'ADAMS']




```python
len(cocs)
```




    1655




```python
conn.executescript('''
DROP TABLE IF EXISTS prep2_2019;
CREATE TABLE prep2_2019 (
    coc_file    VARCHAR,
    region      VARCHAR,
    province    VARCHAR,
    city        VARCHAR,
    cc          VARCHAR,
    bo          VARCHAR,
    v           VARCHAR,
    tot         VARCHAR,
    per         VARCHAR
);
''')
conn.commit()

i = 0
for c in cocs:
    with open(c[0], 'r') as file:
        data = json.load(file)
    for d in data['rs']:
        values = tuple(list(c) + [d['cc'], d['bo'], d['v'], d['tot'], d['per']])
        conn.execute('''INSERT INTO prep2_2019 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', values)
    i+=1
    if i%100==0:
        conn.commit()
        print(datetime.datetime.now().time(), '- Finished', i, 'rows.')
conn.commit()
```

    18:32:10.572379 - Finished 100 rows.
    18:32:11.049598 - Finished 200 rows.
    18:32:11.441989 - Finished 300 rows.
    18:32:11.861411 - Finished 400 rows.
    18:32:12.276664 - Finished 500 rows.
    18:32:12.704843 - Finished 600 rows.
    18:32:13.131572 - Finished 700 rows.
    18:32:13.512401 - Finished 800 rows.
    18:32:13.909030 - Finished 900 rows.
    18:32:14.284618 - Finished 1000 rows.
    18:32:14.645670 - Finished 1100 rows.
    18:32:15.030448 - Finished 1200 rows.
    18:32:15.398708 - Finished 1300 rows.
    18:32:15.783509 - Finished 1400 rows.
    18:32:16.122346 - Finished 1500 rows.
    18:32:16.487400 - Finished 1600 rows.
    


```python
df2 = pd.read_sql('SELECT * FROM prep2_2019', conn)
```


```python
df2.shape
```




    (392551, 9)




```python
df2.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coc_file</th>
      <th>region</th>
      <th>province</th>
      <th>city</th>
      <th>cc</th>
      <th>bo</th>
      <th>v</th>
      <th>tot</th>
      <th>per</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>392546</th>
      <td>/mnt/data/public/elections/nle2019/results/OAV...</td>
      <td>OAV</td>
      <td>EUROPE</td>
      <td>ITALY</td>
      <td>5567</td>
      <td>43710</td>
      <td>6</td>
      <td>11629</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>392547</th>
      <td>/mnt/data/public/elections/nle2019/results/OAV...</td>
      <td>OAV</td>
      <td>EUROPE</td>
      <td>ITALY</td>
      <td>5567</td>
      <td>43711</td>
      <td>6</td>
      <td>11629</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>392548</th>
      <td>/mnt/data/public/elections/nle2019/results/OAV...</td>
      <td>OAV</td>
      <td>EUROPE</td>
      <td>ITALY</td>
      <td>5567</td>
      <td>43712</td>
      <td>1</td>
      <td>11629</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>392549</th>
      <td>/mnt/data/public/elections/nle2019/results/OAV...</td>
      <td>OAV</td>
      <td>EUROPE</td>
      <td>ITALY</td>
      <td>5567</td>
      <td>43713</td>
      <td>15</td>
      <td>11629</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>392550</th>
      <td>/mnt/data/public/elections/nle2019/results/OAV...</td>
      <td>OAV</td>
      <td>EUROPE</td>
      <td>ITALY</td>
      <td>5567</td>
      <td>43714</td>
      <td>1149</td>
      <td>11629</td>
      <td>9.88</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (8843, 7)




```python
df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c_file</th>
      <th>cc</th>
      <th>cn</th>
      <th>ccn</th>
      <th>bo</th>
      <th>bon</th>
      <th>pn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43446</th>
      <td>/mnt/data/public/elections/nle2019/contests/21...</td>
      <td>2187</td>
      <td>MAYOR NCR - TAGUIG CITY</td>
      <td>MAYOR</td>
      <td>6701</td>
      <td>CAYETANO, DIREK LINO (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
    <tr>
      <th>43447</th>
      <td>/mnt/data/public/elections/nle2019/contests/21...</td>
      <td>2187</td>
      <td>MAYOR NCR - TAGUIG CITY</td>
      <td>MAYOR</td>
      <td>6702</td>
      <td>CERAFICA, ARNEL (PDPLBN)</td>
      <td>PARTIDO DEMOKRATIKO PILIPINO LAKAS NG BAYAN</td>
    </tr>
    <tr>
      <th>43448</th>
      <td>/mnt/data/public/elections/nle2019/contests/21...</td>
      <td>2187</td>
      <td>MAYOR NCR - TAGUIG CITY</td>
      <td>MAYOR</td>
      <td>6700</td>
      <td>ANDRADE, SONNY BOY (IND)</td>
      <td>INDEPENDENT</td>
    </tr>
    <tr>
      <th>43449</th>
      <td>/mnt/data/public/elections/nle2019/contests/38...</td>
      <td>3820</td>
      <td>VICE-MAYOR NCR - TAGUIG CITY</td>
      <td>VICE-MAYOR</td>
      <td>10337</td>
      <td>CRUZ, RICARDO JR. (NP)</td>
      <td>NACIONALISTA PARTY</td>
    </tr>
    <tr>
      <th>43450</th>
      <td>/mnt/data/public/elections/nle2019/contests/38...</td>
      <td>3820</td>
      <td>VICE-MAYOR NCR - TAGUIG CITY</td>
      <td>VICE-MAYOR</td>
      <td>10338</td>
      <td>DUEÑAS, JUN (PDPLBN)</td>
      <td>PARTIDO DEMOKRATIKO PILIPINO LAKAS NG BAYAN</td>
    </tr>
  </tbody>
</table>
</div>



### Merge contest and results dataframes


```python
df3 = pd.merge(df2, df)
```


```python
df3.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coc_file</th>
      <th>region</th>
      <th>province</th>
      <th>city</th>
      <th>cc</th>
      <th>bo</th>
      <th>v</th>
      <th>tot</th>
      <th>per</th>
      <th>c_file</th>
      <th>cn</th>
      <th>ccn</th>
      <th>bon</th>
      <th>pn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/mnt/data/public/elections/nle2019/results/REG...</td>
      <td>REGION I</td>
      <td>ILOCOS NORTE</td>
      <td>ADAMS</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>5366</td>
      <td>0.11</td>
      <td>/mnt/data/public/elections/nle2019/contests/1....</td>
      <td>SENATOR PHILIPPINES</td>
      <td>SENATOR</td>
      <td>ABEJO, VANGIE (IND)</td>
      <td>LGBTQ PARTY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/mnt/data/public/elections/nle2019/results/REG...</td>
      <td>REGION I</td>
      <td>ILOCOS NORTE</td>
      <td>BACARRA</td>
      <td>1</td>
      <td>1</td>
      <td>89</td>
      <td>87217</td>
      <td>0.10</td>
      <td>/mnt/data/public/elections/nle2019/contests/1....</td>
      <td>SENATOR PHILIPPINES</td>
      <td>SENATOR</td>
      <td>ABEJO, VANGIE (IND)</td>
      <td>LGBTQ PARTY</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4 = df3[['c_file', 'coc_file', 'region', 'province', 'city', 'cn', 'ccn', 'bon', 'pn', 'v', 'tot', 'per']]
```


```python
df4.to_sql('prep3_2019', conn, if_exists='replace', index=False)
```


```python
df5 = df4.copy()
```


```python
df5 = df5[['region', 'province', 'city', 'ccn', 'bon', 'pn', 'v', 'tot', 'per']]
df5.columns = ['region', 'province', 'city', 'position', 'candidate', 
               'candidate_party', 'votes', 'total_votes', 'percentage']
```

### Convert votes and total_votes to integer, percentage to float


```python
df5.dtypes
```




    region             object
    province           object
    city               object
    position           object
    candidate          object
    candidate_party    object
    votes              object
    total_votes        object
    percentage         object
    dtype: object




```python
df5.votes = df5.votes.astype(int)
df5.total_votes = df5.total_votes.astype(int)
df5.percentage = df5.percentage.astype(float)
df5.percentage = df5.percentage/100
```


```python
df5.head(2)
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
    </tr>
  </tbody>
</table>
</div>



### Load final city-level 2019 election votes data to sqlite3 table and pickle file


```python
df5.to_sql('city_votes_2019', conn, if_exists='replace', index=False)
```


```python
with open('city_votes_2019.pkl', 'wb') as f:
    pickle.dump(df5, f)
```


```python

```
