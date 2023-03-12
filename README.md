# Session-Based-Product-Recommendations


## About
University of Washington MS in Data Science Capstone Project

Project Sponsor - Neiman Marcus

Team Members
- Aamir Darukhanawalla
- Aniket Fadia
- Jessie Ren
- Matthew Chau
- Shweta Mayekar


## Introduction
Neiman Marcus (NM) owns a website where one can shop luxury brand apparels, accessories and more. NM wishes to improve user personalization through product recommendations. Their website has many unregistered visitors, and therefore we implemented session-based recommenders that generate personalized recommendations based on in-session clickstream for unknown visitors. Session-based algorithms rely heavily on the user’s most recent interactions, rather than on their historical preferences. We aim to research and use the best possible session-based approaches to improve product recommendations. We chose GRU4REC, a Recurrent Neural Network based technique specifically designed for session-based recommendation scenarios. 

We train and test GRU4REC on RetailRocket dataset instead of NM data as it was not possible for NM to share their proprietary dataset. NM can use this repository to test the performance of GRU4REC on their data. This model serves as the starting point for using deep learning approaches for Session-Based product recommendations. 

We referred to the following papers for this project -

1. Ludewig, M., & Jannach, D. (2018c). Evaluation of session-based recommendation algorithms. User Modeling and User-Adapted Interaction, 28(4-5), 331–390. https://doi.org/10.1007/s11257-018-9209-6
2. Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS. https://arxiv.org/pdf/1511.06939.pdf
3. Hidasi, B., & Karatzoglou, A. (2018). Recurrent Neural Networks with Top-k Gains for Session-based Recommendations. Proceedings of the 27th ACM International Conference on Information and Knowledge Management. https://doi.org/10.1145/3269206.3271761


## Environment Setup

Create a new conda environment and download the required packages.
```sh
conda env create -f environment.yml
```


## Run instructions
1. Preprocess data for Retail Rocket
   1. Minimum item support – 5 
   2. Minimum session length – 2 
   3. Session threshold - 30 minutes 
   4. Divide data into 5 slices of training and testing sets 
   5. Train data – 25 days 
   6. Test data – 2 days 
   7. Use Session-parallel mini-batches to capture how sessions evolves
```sh
python run_preprocessing.py conf/preprocess/session_based/window/retailrocket.yml
```
2. Train a GRU4REC model
```sh
python run_config.py conf/in conf/out
```


## Results


We train the GRU4REC model with the below hyperparameters -

| Hyperparameter        | Value    |
|-----------------------|----------|
| Loss Function         | Top1-max |
| Activation Function   | Elu-0.5  |
| Dropout Rate          | 0.2      |
| Momentum              | 0.1      |
| Learning Rate         | 0.06     |
| Constrained Embedding | True     |


We test our model on multiple metrics 

| Metrics    | @10    | @20    |
|------------|--------|--------|
| MRR        | 0.3282 | 0.3329 |
| MAP        | 0.3104 | 0.3209 |
| Hit Rate   | 0.5057 | 0.5726 |
| Coverage   | 0.6242 | 0.7939 |
| Popularity | 0.0353 | 0.0320 |































