
# Increasing Neural Network Efficiency Using Neuron Pruning Techniques

## Final Year Project

This repository focuses on improving neural network efficiency using neuron pruning, model compression, and Granger Causality-based neuron importance analysis.

Modern neural networks often contain redundant neurons that increase:

- model size
- parameter count
- inference time
- deployment cost
- memory usage

without providing significant performance improvement.

This project studies how to make neural networks smaller, faster, and more efficient while preserving strong predictive accuracy.

Instead of using only traditional pruning methods like weight magnitude pruning, this project uses a hybrid pruning strategy based on:

- Weight Importance
- Activation Behavior
- Granger Causality Score

to safely identify and remove weak neurons.

The goal is simple:

> Reduce unnecessary complexity without significantly reducing accuracy.

This project is designed as a research-oriented final year project with practical implementation, result analysis, Power BI dashboards, and multi-dataset experimentation.

---

## Main Objective

The primary goal of this project is to improve neural network efficiency by removing low-importance neurons while maintaining model performance.

The system aims to reduce:

- parameter count
- model size
- inference time
- computational cost

while preserving:

- accuracy
- F1 score
- model reliability
- deployment readiness

This helps create models that are better suited for:

- edge devices
- mobile deployment
- real-time systems
- low-resource environments
- production AI systems

---

## Problem Statement

Deep learning models are often over-parameterized.

Many neurons contribute very little to final predictions but still increase:

- training cost
- inference latency
- hardware requirements
- deployment difficulty

Traditional pruning methods often rely only on weight magnitude and may accidentally remove important neurons.

This project solves that problem using Granger Causality-based pruning, where neuron importance is measured more intelligently using both statistical dependency and neural behavior.

---

## Core Idea

The system follows this logic:

```text
Train Model
→ Observe Hidden Neurons
→ Analyze Neuron Relationships
→ Calculate Importance Score
→ Remove Weak Neurons
→ Fine-Tune Smaller Model
→ Compare Performance
````

This creates a smarter pruning pipeline instead of random compression.

---

## Proposed Methodology

### Step 1 — Train Baseline Model

The original neural network is trained normally and used as the reference model.

### Step 2 — Extract Hidden Layer Activations

Neuron activations are recorded to understand how strongly neurons respond during prediction.

### Step 3 — Apply Granger Causality

Granger causality is used to analyze directional dependency between neurons.

This helps answer:

> Does one neuron influence the future behavior of another neuron?

Strong contributors are preserved.

Weak contributors become pruning candidates.

### Step 4 — Calculate Final Importance Score

Each neuron receives an importance score using:

```text
Final Score =
Weight Importance
+ Activation Importance
+ Granger Causality Score
```

This hybrid scoring improves pruning quality.

### Step 5 — Prune Weak Neurons

Low-score neurons are removed using structural pruning.

This actually rebuilds a smaller model.

### Step 6 — Fine-Tune the Pruned Model

The smaller network is retrained to recover performance.

### Step 7 — Performance Evaluation

Baseline and pruned models are compared using:

* Accuracy
* F1 Score
* Test Loss
* Parameter Count
* Model Size
* Inference Time

---

## Current Completed Module

## MNIST — Granger Causality Based Neuron Pruning

The first completed implementation uses the MNIST handwritten digit dataset.

This module includes:

* baseline model training
* activation logging
* Granger causality analysis
* hybrid importance scoring
* structural neuron pruning
* fine-tuning
* result comparison
* Power BI dashboards
* interactive dashboard visualization

This serves as the foundation of the full final-year project.

---

## Planned Extensions

The project is designed to expand across multiple datasets and architectures.

Future planned modules include:

### Car Evaluation Dataset

Tabular dataset pruning for structured data analysis.

### Sonar Dataset

Binary classification pruning for classical ML comparison.

### CIFAR-10 / CIFAR-100

CNN-based pruning for image classification.

### Advanced Architectures

Future extension to:

* CNNs
* Transformer models
* deployment-focused optimization

This makes the project scalable beyond a single dataset.

---

## Power BI Dashboards

Two analytical dashboards are developed for research presentation and decision support.

### Page 1 — Performance Dashboard

Includes:

* Accuracy vs Pruning Percentage
* Parameter Reduction Analysis
* Test Loss vs Pruning
* Inference Time Improvement
* Baseline vs Pruned Comparison
* Model Performance Summary

### Page 2 — Granger Causality Dashboard

Includes:

* Granger Causality Matrix
* Top Influential Neurons
* Pruned Neurons Based on Low Influence
* Statistical Significance Analysis
* Hybrid Importance Score Analysis

These dashboards help convert raw ML outputs into clear research insights.

---

## Repository Structure

```text
neural-network-efficiency-pruning/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── mnist-granger-pruning/
│   ├── README.md
│   ├── run_pipeline.py
│   ├── requirements.txt
│   ├── src/
│   ├── data/
│   ├── notebooks/
│   ├── results/
│   ├── dashboards/
│   └── docs/
```

Each module is maintained independently for clean development, testing, and presentation.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Statsmodels
* Google Colab
* Power BI
* GitHub

---

## Key Research Outcome

The project demonstrates that neural networks can be significantly compressed while maintaining near-baseline performance.

Results show:

* strong parameter reduction
* improved inference speed
* reduced model size
* minimal accuracy loss
* better deployment efficiency

This proves that selective neuron pruning is more effective than simple weight-based pruning.

---

## Future Scope

Future improvements include:

* pruning on larger datasets
* CNN and Transformer pruning
* deployment on edge devices
* automated pruning threshold optimization
* integration with MLOps pipelines
* medical image model compression
* real-time AI system optimization

This project can be extended into publication-level research.

---

## Conclusion

Granger Causality-based neuron pruning provides a smarter alternative to traditional pruning methods.

By identifying directional neuron dependencies and combining them with activation and weight analysis, the system removes weak neurons while preserving strong contributors.

The result is:

> smaller, faster, and more efficient neural networks without major performance loss

This improves real-world deployment readiness and creates strong research value for modern AI systems.

---

## Authors

*Srusti 1NT23AD052
*Sri Nidhi Shinde 1NT23AD050
*Sumangala Vastrad 1NT24AD404
*Vinay V 1NT23AD052

### Under the guidance of :
Archana Mathur
Professor

Department of Artificial Intelligence and Data Science

---

