# Clustered-FL-BrainAGE
Official implementation of paper ["Brain Age Estimation Using Structural MRI: A Clustered Federated Learning Approach"](https://ieeexplore.ieee.org/document/10189329), Accepted to [IEEE COINS 2023](https://coinsconf.com/) conference.

## Abstract
Estimating brain age based on structural Magnetic Resonance Imaging (MRI) is one of the most challenging and prominent areas of research in recent medical imaging and neuroscience studies. The significance of brain age prediction in the early diagnosis of neurological disorders has fueled a resurgence of interest in this field. Various studies have addressed this issue using a spectrum of techniques, from traditional machine learning to deep neural networks. The majority of these techniques employ centralized paradigms, which do not adequately preserve privacy. To tackle this problem, a handful of studies have utilized a federated approach. In this study, we propose a novel hierarchical clustered federated learning approach that carefully captures and considers the similarities of the clients' predictions on a certain benchmark dataset. This method enhances performance in Non-Independent and Identically Distributed (Non-IID) environments while preserving privacy. We use a multi-site dataset that provides a broad variety of MRI scans, characterizing a proper Non-IID environment. Our method achieves a Mean Absolute Error (MAE) of 3.86, while the non-clustered FedAvg federated approach attains a 4.14 MAE on the test set.

## Setup & Usage
1. Clone the repository.
   ```
   git clone https://github.com/Abtinmy/Clustered-FL-BrainAGE.git
   ```
2. Create a conda environment using the `requirements.txt` file.
   ```
   conda create --name <env> --file requirements.txt
   ```
3. Run the sample experiment using the shell command in the `script` directory. Arguments and their descriptions are explained in `arguments_parser.py`.
   ```
   cd scripts/
   sh sample.sh
   ```
## Citation
```
@inproceedings{cheshmi2023brain,
  title={Brain Age Estimation Using Structural MRI: A Clustered Federated Learning Approach},
  author={Cheshmi, Seyyed Saeid and Mahyar, Abtin and Soroush, Anita and Rezvani, Zahra and Farahani, Bahar},
  booktitle={2023 IEEE International Conference on Omni-layer Intelligent Systems (COINS)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement
This project is inspired by [FLIS](https://github.com/MMorafah/FLIS) repository.
