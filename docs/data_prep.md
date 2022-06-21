### Data Preparation Stage 

- convert the date into train.tsv and test.tsv in 70:30 ratio
```
data.xml
    |-tran.tsv
    |-train.tsv
    
````

- We are choosing only three tags in the xml data -1. row Id, 2. title and body, 3. Tags (Stackoverflow tags specific to python)

|Tags|Feature Names|
|-|-|
|row Id|row Id|
|test and body|test and body|
|Stackoverflow tags|Label - python|

