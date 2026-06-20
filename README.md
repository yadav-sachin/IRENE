<img src="media/KDD_logo.png" height="80" align="right"/>
<img src="media/microsoft_logo.png" height="80" align="right"/>

# IRENE: Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval

[![KDD 2024](https://img.shields.io/badge/Venue-KDD%202024-blue)](https://dl.acm.org/doi/10.1145/3637528.3672046)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](http://manikvarma.org/pubs/yadav24.pdf)
[![Supplementary](https://img.shields.io/badge/Supplementary-PDF-orange)](https://www.siddarthasokan.com/assets/pdf/EMMETT_IRENE24_Supp.pdf)

Official implementation of **"Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval"** (KDD 2024).

---

## Overview

**EMMETT** is the general algorithmic framework introduced in this paper for synthesizing classifiers on-the-fly for novel (zero-shot) items, by leveraging classifiers already learned for observed items during training. **IRENE** is a simple and scalable instance of EMMETT, comprising two components:

- **Classifier Selector (S):** Retrieves *K* nearest neighbor labels from the seen set for a given novel label, using an ANNS index over label embeddings.
- **Meta-Classifier Generator (G):** A Transformer encoder that takes the novel label's embedding and the classifiers of its *K* neighbors as input, and produces a meta-classifier for retrieval.

IRENE is plug-and-play atop any dense retriever and improves P@1 by **+10.1%** and R@10 by **+11.9%** on average across datasets in the zero-shot setting.

---

## Installation

```bash
conda create -n irene python=3.8 -y
conda activate irene
pip install -r requirements.txt
```

---

## Data Preparation

### Zero-Shot Dataset Splits

Datasets are sourced from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) (LF-prefixed variants with label text features). 10% of labels are randomly drawn from the full label set and designated as **novel**; the remaining 90% are **seen**:

```
All labels  ──────────────────────────────────────────
            │◄────── 90% seen ────────►│◄─ 10% novel ─►│

Training    queries with ≥1 seen-label positive  →  Y.trn.npz      [N_trn   × L_seen]
Zero-shot   test queries with ≥1 novel positive  →  Y.tst_zero.npz [N_zero  × L_novel]
Generalized all test queries, all labels         →  Y.tst_full.npz [N_test  × L_all]
```

The (query, label) pairs in each split come directly from the original dataset's ground-truth annotations — no new pairs are synthesized. For the zero-shot test, only pairs whose label falls in the novel set are retained; for the generalized test, all original test pairs are kept across the full label space. The model is trained exclusively on seen labels and evaluated at test time on novel labels it has never encountered. Filter files follow the [XMLRepository reciprocal-pair convention](http://manikvarma.org/downloads/XC/XMLRepository.html#filter) to exclude trivially self-referential predictions from evaluation.

### Directory Structure

Training requires precomputed embeddings and classifiers from a base extreme classifier. All embeddings must be in **un-normalized** form — IRENE applies normalization internally. Organize files as:

```
Datasets/
└── <Dataset>/
    ├── Y.trn.npz                     # train relevance matrix           [N_trn   × L_seen]
    ├── Y.tst_zero.npz                # zero-shot test relevance         [N_zero  × L_novel]
    ├── Y.tst_full.npz                # generalized test relevance       [N_test  × L_all]
    ├── filter_labels_test_zero.txt
    └── filter_labels_test_full.txt

Dataset_Assets/
└── <Dataset>/<Base Retriever>/
    │
    │   # Query embeddings (un-normalized)
    ├── trn_X_unnorm.npy              #  train queries                   [N_trn   × D]
    ├── tst_X_zero_unnorm.npy         #  zero-shot test queries          [N_zero  × D]
    ├── tst_X_full_unnorm.npy         #  generalized test queries        [N_test  × D]
    │
    │   # Label embeddings (un-normalized)
    ├── Y_trn_unnorm.npy              #  seen labels                     [L_seen  × D]
    ├── Y_zero_unnorm.npy             #  novel labels                    [L_novel × D]
    ├── Y_full_unnorm.npy             #  all labels  = concat(trn, zero) [L_all   × D]
    │
    │   # Per-label classifiers from base retriever (un-normalized)
    ├── Y_trn_classifiers_unnorm.npy  #  seen label classifiers          [L_seen  × D]
    │
    │   # k-NN neighbors in seen label space (computed via ANNS over Y_trn_unnorm)
    ├── Y_trn_neighbor_indices.npy    #  seen label → seen neighbors     [L_seen  × K_max]
    ├── Y_trn_neighbor_scores.npy     #  neighbor cosine similarities    [L_seen  × K_max]  (int, 0–10)
    └── Y_zero_neighbor_indices.npy   #  novel label → seen neighbors    [L_novel × K_max]
```

> **Note on neighbor scores:** The cosine similarity between a label and each of its k-NN neighbors is discretized into an integer in [0, 10] and stored in `Y_trn_neighbor_scores.npy`. IRENE maps these integers to learned `ScoreEmbeddings` that are added to each neighbor's classifier before it enters the Transformer, giving the model a proximity-aware signal over the neighborhood. Novel label neighbors (`Y_zero`) are not scored — a uniform score of 1 is used at inference.

---

## Training

### Step 1 — Train Base XC Classifier

Train a base encoder on the seen labels using a one-vs-all BCE loss to produce per-label embeddings and classifiers, then compute k-NN neighbors in label space. Base classifiers used in this work:

| Encoder | Reference |
|---------|-----------|
| [NGAME](https://github.com/Extreme-classification/NGAME) | Dahiya et al., WSDM 2023 |
| [ANCE](https://github.com/microsoft/ANCE) | Xiong et al., ICLR 2021 |
| [DPR](https://github.com/facebookresearch/DPR) | Karpukhin et al., EMNLP 2020 |
| MACLR | Xiong et al., arXiv 2021 |

Place outputs under `Dataset_Assets/<Dataset>/<Base Retriever>/` as above.

### Step 2 — Train IRENE

Training is driven by YAML configs. See [`configs/base_config.yaml`](configs/base_config.yaml) for all parameters and [`configs/<Dataset>/meta_clf_gen.yaml`](configs/LF-AmazonTitles-1.3M_10/meta_clf_gen.yaml) for dataset-specific settings.

```bash
dataset="LF-WikiHierarchy-550K_10"
python train.py configs/${dataset}/meta_clf_gen.yaml
```

**All datasets:**
```bash
python train.py configs/LF-AOL-270K_10/meta_clf_gen.yaml
python train.py configs/LF-WikiHierarchy-550K_10/meta_clf_gen.yaml
python train.py configs/LF-AmazonTitles-1.3M_10/meta_clf_gen.yaml
python train.py configs/LF-Wikipedia-500K_10/meta_clf_gen.yaml
```

**Multi-GPU clustering:**
```bash
python train.py configs/${dataset}/meta_clf_gen.yaml --cls_devices "0,1,2,3"
```

### Outputs

Training writes to `Results/<project>/<dataset>/<base_retriever>/<expname>/`:

```
Results/
└── <project>/<dataset>/<base_retriever>/<expname>/
    ├── log.txt                            # loss and metrics at each eval epoch
    ├── state_dict_ep_-1.pt               # checkpoint before training
    ├── state_dict_ep_{0,10,20,30,39}.pt  # checkpoint at each eval epoch
    ├── state_dict.pt                     # final checkpoint
    └── embeddings/
        ├── Y_zero_{epoch}.irene.npy      # IRENE label representations for novel labels
        └── Y_full_{epoch}.irene.npy      # IRENE label representations for all labels
```

### Key Hyperparameters

The two most important IRENE-specific hyperparameters, validated through ablations (Table 4 in the paper):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_neighbors` (K) | 3 | Smaller K yields tighter generalization bound and better empirical performance; K=3 works well across all datasets |
| `num_layers` (D) | 1 | One Transformer layer is sufficient; deeper G tends to overfit |
| `neighbor_itself` | `False` | Whether a seen label attends to itself during training. Must be `False`: at test time, novel labels are absent from the seen label space and can never be their own neighbor, so including self-attention during training would create a train/test mismatch |

---

## Results

Datasets are from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), with a 90/10 seen/novel split of labels. Metrics: **P@1** and **R@10**.

### Zero-Shot Retrieval

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">LF-AOL-270K</th>
      <th colspan="2">LF-WikiHierarchy-550K</th>
      <th colspan="2">LF-AmazonTitles-1.3M</th>
      <th colspan="2">LF-Wikipedia-500K</th>
    </tr>
    <tr>
      <th>P@1</th><th>R@10</th>
      <th>P@1</th><th>R@10</th>
      <th>P@1</th><th>R@10</th>
      <th>P@1</th><th>R@10</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>NGAME</td><td>30.90</td><td>54.20</td><td>46.01</td><td>58.66</td><td>30.42</td><td>36.44</td><td>46.96</td><td>65.27</td></tr>
    <tr><td><b>NGAME + IRENE</b></td><td><b>36.47</b></td><td><b>59.57</b></td><td><b>69.29</b></td><td><b>80.40</b></td><td><b>31.56</b></td><td><b>38.83</b></td><td>44.91</td><td>67.79</td></tr>
    <tr><td>ANCE</td><td>33.43</td><td>67.84</td><td>43.06</td><td>56.28</td><td>22.38</td><td>30.72</td><td>30.67</td><td>58.91</td></tr>
    <tr><td><b>ANCE + IRENE</b></td><td><b>36.84</b></td><td><b>67.82</b></td><td><b>66.54</b></td><td><b>82.10</b></td><td><b>22.75</b></td><td><b>32.72</b></td><td><b>41.59</b></td><td><b>71.59</b></td></tr>
    <tr><td>DPR</td><td>30.38</td><td>53.82</td><td>44.84</td><td>59.29</td><td>31.10</td><td>40.98</td><td>42.90</td><td>71.20</td></tr>
    <tr><td><b>DPR + IRENE</b></td><td><b>36.80</b></td><td><b>60.22</b></td><td><b>69.65</b></td><td><b>80.01</b></td><td>30.49</td><td>40.31</td><td>42.19</td><td>70.50</td></tr>
    <tr><td>MACLR</td><td>11.31</td><td>18.24</td><td>30.37</td><td>35.47</td><td>21.93</td><td>28.59</td><td>39.56</td><td>68.53</td></tr>
    <tr><td><b>MACLR + IRENE</b></td><td><b>34.32</b></td><td><b>61.29</b></td><td><b>69.45</b></td><td><b>81.43</b></td><td>21.56</td><td>28.77</td><td><b>44.64</b></td><td><b>73.05</b></td></tr>
    <tr><td>SemSup-XC</td><td>26.27</td><td>36.31</td><td>57.45</td><td>46.81</td><td>11.28</td><td>11.68</td><td>46.60</td><td>57.08</td></tr>
    <tr><td><a href="https://github.com/Extreme-classification/DEXA">DEXA</a></td><td>21.68</td><td>41.85</td><td>54.83</td><td>66.89</td><td>28.83</td><td>35.19</td><td>42.76</td><td>67.37</td></tr>
  </tbody>
</table>

### Generalized Zero-Shot Retrieval

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">LF-AOL-270K</th>
      <th colspan="2">LF-WikiHierarchy-550K</th>
      <th colspan="2">LF-AmazonTitles-1.3M</th>
      <th colspan="2">LF-Wikipedia-500K</th>
    </tr>
    <tr>
      <th>P@1</th><th>R@10</th>
      <th>P@1</th><th>R@10</th>
      <th>P@1</th><th>R@10</th>
      <th>P@1</th><th>R@10</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>NGAME</td><td>20.16</td><td>38.27</td><td>66.19</td><td>27.08</td><td>45.14</td><td>30.25</td><td>81.86</td><td>69.58</td></tr>
    <tr><td><b>NGAME + IRENE</b></td><td><b>35.11</b></td><td><b>52.30</b></td><td><b>91.33</b></td><td><b>40.09</b></td><td><b>47.77</b></td><td><b>31.49</b></td><td>78.99</td><td>69.27</td></tr>
    <tr><td>ANCE</td><td>22.63</td><td>49.72</td><td>68.76</td><td>25.89</td><td>27.65</td><td>17.31</td><td>42.91</td><td>43.39</td></tr>
    <tr><td><b>ANCE + IRENE</b></td><td><b>30.84</b></td><td><b>51.75</b></td><td><b>90.72</b></td><td><b>39.74</b></td><td><b>36.78</b></td><td><b>22.34</b></td><td><b>71.39</b></td><td><b>63.46</b></td></tr>
    <tr><td>DPR</td><td>19.71</td><td>37.99</td><td>65.19</td><td>26.73</td><td>38.18</td><td>26.93</td><td>51.54</td><td>61.84</td></tr>
    <tr><td><b>DPR + IRENE</b></td><td><b>35.07</b></td><td><b>52.57</b></td><td><b>89.52</b></td><td><b>39.84</b></td><td><b>43.08</b></td><td><b>29.25</b></td><td><b>70.39</b></td><td><b>66.71</b></td></tr>
    <tr><td>MACLR</td><td>9.26</td><td>7.52</td><td>59.44</td><td>14.31</td><td>27.50</td><td>15.99</td><td>46.59</td><td>46.62</td></tr>
    <tr><td><b>MACLR + IRENE</b></td><td><b>30.40</b></td><td><b>44.71</b></td><td><b>88.81</b></td><td><b>38.37</b></td><td><b>31.49</b></td><td><b>18.85</b></td><td><b>70.52</b></td><td><b>62.82</b></td></tr>
    <tr><td>SemSup-XC</td><td>26.12</td><td>23.92</td><td>90.51</td><td>28.37</td><td>25.13</td><td>15.21</td><td>54.20</td><td>38.08</td></tr>
    <tr><td><a href="https://github.com/Extreme-classification/DEXA">DEXA</a></td><td>25.09</td><td>46.76</td><td>76.18</td><td>36.17</td><td>48.19</td><td>30.89</td><td>67.98</td><td>65.86</td></tr>
  </tbody>
</table>

> Full results (P@1, P@3, P@5, R@3, R@5, R@10, R@30, R@100) are in Tables 14–15 of the [supplementary document](https://www.siddarthasokan.com/assets/pdf/EMMETT_IRENE24_Supp.pdf).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yadav2024extrememetaclassification,
  author    = {Yadav, Sachin and Saini, Deepak and Buvanesh, Anirudh and Paliwal, Bhawna
               and Dahiya, Kunal and Asokan, Siddarth and Prabhu, Yashoteja
               and Jiao, Jian and Varma, Manik},
  title     = {Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series    = {KDD '24},
  year      = {2024},
  doi       = {10.1145/3637528.3672046}
}
```
