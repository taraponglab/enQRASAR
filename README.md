# Predict Skin Irritation based on pIC50 using command-line tool application

Author: Tarapong Srisongkram

## Description

This is a simple command-line tool to load and use a pre-trained EnRAQSAR model on your local computer. With this tool, you can make skin irritation predictions directly from the terminal without the need for a GUI or web interface. This software is based on the paper: **Ensemble Read-Across Quantitative Structure-Activity Relationship Algorithm for Predicting Skin Irritation Toxicity (Submitted)**

## Prerequisites

1. **Python**: Ensure you have Python 3.9 or newer installed on your machine.

2. **Required Libraries**: Before you start, make sure to install the required Python libraries by running (in terminal):

First locate to the path of this software
```bash
    cd path of the software
```
Then install the required Python libraries

```bash
    pip install -r requirements.txt
```

3. **Input File**: This software accepts input as an **Excel file**. The format of Excel should include LigandID as index column, canonical_smiles as the SMILES column.
for example.

|  LigandID |  canonical_smiles |
|---|---|
|  Ciproflixacin |  O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O |

The test of the compound is unlimited. You can put the compound with SMILES as much as you want. Then, place the input file in the folder **input**

## Usage

## Basic Command:
Analyze the chemicals in the `INPUT_FILE_NAME` in the input folder and save in the output folder as CSV file.
```Bash
python main.py --input INPUT_FILE_NAME
```

## Options
`--input` : This option is available for specifies the input file name `INPUT_FILE_NAME`, default is `example.xlsx`. 

Default will return the output from the example.xlsx file 
```Bash
python main.py
```
Specify the `INPUT_FILE_NAME`
```Bash
python main.py --input INPUT_FILE_NAME
```
`--output`: This option is available for specifies the output file name `OUTPUT_FILE_NAME`. The default is `output.csv`, and this file will save in the `output` folder

```Bash
python main.py --input INPUT_FILE_NAME --output OUTPUT_FILE_NAME
```

# Examples

1. Predict skin irritation of Ciprofloxacin and save in output.csv:

```Bash
python main.py --input ciplofoxacin.xlsx
```
2. Predict skin irritation of Ciprofloxacin and save in result.csv:

```Bash
python main.py --input ciplofoxacin.xlsx --ouput result.csv
```

The result is shown below:

| LigandID  | QSAR | ECFP | APF | Predicted pIC50  | Similarity  |
|---|---|---|---|---|---|
|  Ciproflixacin  | 4.03  | 4.06  | 4.38  | 4.01  | 0.64  |

# Interpretation

`LigandID` : Chemical Name \
`QSAR` : Physicochemical properties-based QSAR predictive result \
`ECFP` : Extended circular fingerprint (ECFP)-based Read-Across predictive result \
`APF`  : Atom Pairs 2D fingerprint (APF)-based Read-Across predictive result \
`Predicted pIC50` : Stacked ensemble of QSAR, ECFP, APF based prediction (final pIC50 result). High value (pIC50 > 5) indicates skin irritation chemical \
`Similarity` : Tanimoto-based similarity to the training data set (excluded outliers)

# Contributing
If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request.

# License
This project is licensed under the MIT License.

# Feedback and Issues
If you have feedback or run into any issues, please submit an issue on the project's GitHub page.