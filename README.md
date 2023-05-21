# MFIOD
The code and dataset for paper "Fusing multi-scale fuzzy information to detect outliers".

## Datasets
We use 9 public datasets to assess the model performances. The details of the datasets are provided in below table: 


| No |              Datasets              | Abbreviation | #Attributes | #Outliers | #Objects | Outlier ratio |
|:--:|:----------------------------------:|:------------:|:-----------:|:---------:|:--------:|:-------------:|
|  1 |              Diabetes              |     Diab     |      8      |     26    |    526   |     4.94%     |
|  2 |             Ionosphere             |     Iono     |      34     |     24    |    249   |     9.64%     |
|  3 |          Cardiotocography          |     Card     |      21     |     33    |   1688   |     1.95%     |
|  4 |                Pima                |     Pima     |      9      |     55    |    555   |     9.91%     |
|  5 |                Sonar               |     Sonar    |      60     |     10    |    107   |     9.35%     |
|  6 | Wisconsin diagnostic breast cancer |     Wdbc     |      31     |     39    |    396   |     9.85%     |
|  7 |             Page blocks            |     Page     |      10     |    258    |   5171   |     4.99%     |
|  8 |       Wisconsin breast cancer      |      Wbc     |      9      |     39    |    483   |     8.07%     |
|  9 |                Yeast               |     Yeast    |      8      |     5     |   1141   |     0.44%     |

## Environment
* numpy=1.23.5
* python=3.8.16
* scikit-learn=1.2.0
* scipy=1.9.3
