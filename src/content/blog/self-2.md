---
author: Elouan Gardès
pubDatetime: 2024-05-23T15:22:00Z
#modDatetime: 2023-12-21T09:12:47.400Z
title: testtttt
slug: test
featured: true
draft: false
tags:
  - representation-learning
  - contrastive-learning
  - predictive-coding
  - transfer-learning
  - time-series
  - medical-data
description:
  Let's explore and compare some interesting approaches to representation learning of medical time-series! Two datasets, a thrilling battle between models and a fast predictive coding implementation for contrastive learning of time-series.
---

## Pneumonia Prediction Dataset

### Supervised Model for Transfer
We use our simple Residual CNN from part 1. It achieved strong performance on the PTB Diagnostic dataset with a very low parameter count, making it suitable for the self-supervised learning task in question 2. We train it for 5 epochs (and we will continue to train for 5 epochs across all experiments on the MIT-BIH dataset for fairness). We report the performance from straightforward training. Additionally, we provide the performance obtained when class weights are considered in the loss function. In other words, we use the proportion of each class in the dataset to weigh the loss.

#### Performance Metrics

| **Accuracy** | **F1** | Precision - weighted | Precision - macro | Recall - macro | **OVR** |
|--------------|--------|----------------------|-------------------|----------------|---------|
| **0.97**     | **0.96** | **0.97**            | **0.93**          | 0.76           | **0.97**|
| 0.88         | 0.90   | 0.95                 | 0.62              | **0.90**       | **0.98**|

#### Figures
![Figure 1](../../assets/images/time-series/1.png)
![Figure 2](../../assets/images/time-series/2.png)