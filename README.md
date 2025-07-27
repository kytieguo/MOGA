## Drug response prediction--MOGA

### Quick start：

Run `./train.py` predict drug response.

```
python train.py --learning_rate 0.001 --out_size 128 --att_head 3
```

We can download the source code and data using Git.

```
git clone https://github.com/kytieguo/MOGA.git
```
### Dataset 
* [cell\_line\_copynumber.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_copynumber.csv "cell_line_copynumber.csv"): Copy number variations data.
* [cell\_line\_data.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_data.csv "cell_line_data.csv"): List of cell line data.
* [cell\_line\_expression.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_expression.csv "cell_line_expression.csv"): Transcriptomics data.
* [cell\_line\_metabolomic.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_metabolomic.csv "cell_line_metabolomic.csv"): Metabolomics data.
* [cell\_line\_methylation.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_methylation.csv "cell_line_methylation.csv"): DNA methylation data.
* [cell\_line\_mutation.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_mutation.csv "cell_line_mutation.csv"): Mutations data.
* [cell\_line\_protein.csv](https://github.com/kytieguo/MOGA/blob/master/data/cell_line_protein.csv "cell_line_protein.csv"): Proteomics data.
* [drug.csv](https://github.com/kytieguo/MOGA/blob/master/data/drug.csv "drug.csv"): List of drugs data.
* [nosen\_data.csv](https://github.com/kytieguo/MOGA/blob/master/data/nosen_data.csv "nosen_data.csv"): None sensitive data.
* [response.csv](https://github.com/kytieguo/MOGA/blob/master/data/response.csv "response.csv"): Response data after processing.
* [sen\_data.csv](https://github.com/kytieguo/MOGA/blob/master/data/sen_data.csv "sen_data.csv"): Sensitive data.

### Envs
| Name| Version |
| --- | --- |
|python  |3.8.10  | 
|dgl|2.4.0+cu118  |
|deepchem|2.8.0  |
|numpy| 1.23.5 |
|networkx|3.1  |
|pandas|2.0.3  |
|scipy|1.10.1  |
|scikit-learn|1.3.2  |
|torch|2.4.0+cu118  |
|torchaudio|2.4.0+cu118  |
|torchvision|0.19.0+cu118  |
