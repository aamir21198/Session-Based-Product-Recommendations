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
**Neiman Marcus** (NM) is a Fashion Retailer selling luxury apparels & accessories via their proprietary website. Their website has many unregistered visitors, and they wish to improve personalized recommendations for these users. Recommending the right product helps cultivate brand loyalty, stimulates more site visits, and encourages more interactions with the brand. We implemented session-based recommenders that generate personalized recommendations based on in-session clickstream for unknown visitors. We implemented session-based recommender that relies heavily on the user’s most recent interactions rather than on their historical preferences. We aim to research and use the best possible session-based approaches to improve product recommendations. 

For building our model, we referenced the research paper ‘Evaluation of Session-based Recommendation Algorithms’ by Malte Ludewig and Dietmar Jannach [1]. We also referenced other papers [2], [3].

We chose GRU4REC, a complex Recurrent Neural Network based technique specifically designed for session-based recommendation scenarios. Our experiments revealed that GRU4REC performed suitably well for product recommendation. 

We train and test GRU4REC on RetailRocket dataset instead of NM data as it was not possible for NM to share their proprietary dataset. NM can use this repository to test the performance of GRU4REC on their data. This model serves as the starting point for using deep learning approaches for Session-Based product recommendations.

## Data
Our [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) dataset comes from Kaggle. It has 1,048,575 rows and 234,838 products.

The fields in this dataset include
* Timestamp
* Session ID
* Product ID
* Event (View / AddToCart)

## Model Selection - GRU4REC
Gru4Rec is a Recurrent Neural Network that predicts the next viewed product based on the current session. The session can be represented by an actual product or a weighted sum of seen events. While the usage of RNNs for session-based, or more generally, sequential prediction problems is a natural choice, this particular network architecture, the choice of the loss functions, and the use of session-parallel mini-batches to speed up the training phase are key innovative elements of Gru4Rec approach.

The system uses gradient recurrent unit (GRU) layers at its core, and feedforward layers at the end to provide output. To order sessions, it uses session-parallel mini-batches, where the first event of the first several sessions forms the input of the first mini-batch, and so on. The system samples items based on their popularity, and negative samples come from other training mini-batch examples. One of the choices of the loss functions is TOP1, which is a ranking loss that approximates the relative rank of an item with a regularization term added to ensure that negative sample scores are close to zero.

## Environment Setup

Create a new conda environment and download the required packages.
```sh
conda env create -f environment.yml
```


## Run instructions
1. Preprocess data for RetailRocket
   1. Discard products with less than 6 occurrences
   2. Discard sessions with less than 3 products
   3. Divide user-activity into sessions based on a 30-minute threshold of inactivity
   4. Split data into 5 slices of training and testing sets for each 5-month period. 
   5. For each slice:
      1. Train data - 25 days
      2. Test data - 2 days 
   6. Use Session-parallel mini-batches to capture how sessions evolves

```sh
python run_preprocessing.py conf/preprocess/session_based/window/retailrocket.yml
```

2. Train a GRU4REC model
```sh
python run_config.py conf/in conf/out
```

## Approach
*  We split the data into 5 slices of equal size in days (25 days for Training, subsequent 2 days for Testing). The 5 data slices allowed us to ensure multiple measurements of evaluation metrics with different test sets.
* We implemented session-parallel mini-batch scheme within the Gru4Rec network as explained in the reference paper by Ludewig - 'Evaluation of session-based recommendation algorithms' [1]. The input of the network is formed by a single product, which is one-hot encoded in a vector representing the entire product space, and the output is a vector of similar shape that should give a ranking distribution for the subsequent product.
* While training and predicting with the help of this network architecture, the products of a session are fed into the network in the correct order and the hidden state of the GRUs is reset after a session ends.

## HyperParameter Tuning

We tried various combinations of hyperparameter values and obtained optimal results using the following hyperparameters -

| Hyperparameter        | Value    |
|-----------------------|----------|
| Loss Function         | TOP1-max |
| Activation Function   | Elu-0.5  |
| Dropout Rate          | 0.2      |
| Momentum              | 0.1      |
| Learning Rate         | 0.06     |
| Constrained Embedding | True     |

## Results

We test our model on multiple metrics. Below, we compare our Gru4Rec model results with the benchmark model from Ludewig’s research paper [1]

| Metrics       | Our model | Benchmark Model |
|---------------|-----------|-----------------|
| MRR@20        | 0.332     | 0.243           |
| Hit Rate@20   | 0.573     | 0.480           |
| Coverage@20   | 0.794     | 0.602           |
| Popularity@20 | 0.032     | 0.060           |

* We reported four metrics computed from the top 20 recommended products.
* We reported the average of the metrics for all 5 slices.
* MRR (Mean Reciprocal Rank) evaluates to what extent can the immediate next product in a session be predicted.

## Conclusion
* We found a combination of optimal hyperparameter values that maximized our model's performance thereby improving the results found in Ludewig and Jannach’s research paper 
* This model can be utilized by Neiman Marcus along with their existing recommender models to improve personalization and product recommendations for unregistered users.

## References
* [1] Ludewig, M., & Jannach, D. (2018c). Evaluation of session-based recommendation algorithms. User Modeling and User-Adapted Interaction, 28(4-5), 331–390. https://doi.org/10.1007/s11257-018-9209-6

* [2] Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS. https://arxiv.org/pdf/1511.06939.pdf

* [3] Hidasi, B., & Karatzoglou, A. (2018). Recurrent Neural Networks with Top-k Gains for Session-based Recommendations. Proceedings of the 27th ACM International Conference on Information and Knowledge Management. https://doi.org/10.1145/3269206.3271761
