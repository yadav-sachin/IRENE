<img src="media/KDD_logo.png" height="40" align="right"/>

# IRENE
This is the official codebase for KDD 2024 paper Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval.

## Training IRENE 
The codebase assumes following data structure: <br>
```
Datasets/
└── LF-AmazonTitles-1.3M # Dataset name
    ├── raw
    │   ├── trn_X.txt # train input file, ith line is the text input for ith train data point
    │   ├── tst_X.txt # test input file, ith line is the text input for ith test data point
    │   └── Y.txt # label input file, ith line is the text input for ith label in the dataset
    ├── Y.trn.npz # train relevance matrix (stored in scipy sparse npz format), num_train x num_seen_labels
    └── Y.tst_zero.npz # zero-shot test relevance matrix (stored in scipy sparse npz format), num_zero_test x num_novel_labels
    └── Y.tst_full.npz # generalized zero-shot test relevance matrix (stored in scipy sparse npz format), num_full_test x (num_seen_labels + num_novel_labels)
```