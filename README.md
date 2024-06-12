<img src="media/KDD_logo.png" height="80" align="right"/>

# IRENE
This is the official codebase for KDD 2024 paper Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval.

## Training IRENE 
The codebase assumes following data structure: <br>
```
Datasets/
└── LF-AmazonTitles-1.3M # Dataset name
    ├── bert-base-uncased-32
    │   ├── trn_X_input_ids.dat
    │   ├── tst_X_input_ids.dat
    │   ├── Y_trn_input_ids.dat
    │   ├── Y_zero_input_ids.dat
    │   ├── Y_full_input_ids.dat
    ├── raw
    │   ├── trn_X.txt # train input file, ith line is the text input for ith train data point
    │   ├── tst_X_zero.txt # test input file, ith line is the text input for ith test data point
    │   ├── Y_trn.txt # labels input file, ith line is the text input for the ith zero-shot label
    │   ├── Y_zero.txt # labels input file, ith line is the text input for the ith zero-shot label
    │   └── Y_full.txt # label input file, ith line is the text input for ith label in the dataset
    ├── Y.trn.npz # train relevance matrix (stored in scipy sparse npz format), num_train x num_seen_labels
    └── Y.tst_zero.npz # zero-shot test relevance matrix (stored in scipy sparse npz format), num_zero_test x num_novel_labels
    └── Y.tst_full.npz # generalized zero-shot test relevance matrix (stored in scipy sparse npz format), num_full_test x (num_seen_labels + num_novel_labels)
    └── filter_labels_test_zero.txt
    └── filter_labels_test_full.txt
```