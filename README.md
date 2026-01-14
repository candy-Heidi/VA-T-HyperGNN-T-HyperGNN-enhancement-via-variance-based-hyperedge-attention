# VA-T-HyperGNN: T-HyperGNN Enhancement via Variance-based Hyperedge Attention

This repository accompanies a research paper that proposes  
**VA-T-HyperGNN, an enhanced Hypergraph Neural Network model that improves T-HyperGNN using variance-based hyperedge attention**.

The code in this repository is provided for reference only.  
This README summarizes the **research motivation, proposed method, and experimental findings** of the paper.

---

## 📌 Research Motivation

Standard Graph Neural Networks (GNNs) are effective at modeling pairwise relationships  
but struggle to represent **high-order group interactions** involving more than two entities.

Hypergraphs address this limitation by allowing a single **hyperedge to connect multiple nodes**,  
making them suitable for modeling complex group relationships such as co-authorship or co-citation networks.

Among Hypergraph Neural Networks, **T-HyperGNN** models high-order relationships using tensor-based message passing.  
However, T-HyperGNN assigns hyperedge importance based solely on **hyperedge cardinality (number of nodes)**  
and ignores whether the connected nodes are **semantically coherent or heterogeneous**.

---

## 🔍 Key Idea

This paper introduces **VA-T-HyperGNN**, which enhances T-HyperGNN by incorporating  
**variance-based attention over hyperedges**.

The core idea is:

- Hyperedges with **low variance** among node features are likely to represent **semantically coherent groups**
- Hyperedges with **high variance** are more likely to contain noisy or heterogeneous information

Therefore, hyperedges are weighted based on the **variance of node features inside each hyperedge**,  
allowing the model to focus on meaningful group interactions.

---

## 🧠 Method Overview

VA-T-HyperGNN extends the original T-HyperGNN framework by adding a parallel attention pathway:

1. **Standard T-HyperGNN Message Passing**
   - Computes tensor-based hyperedge messages as in the original model

2. **Variance-based Hyperedge Attention**
   - Computes the centroid of node features within each hyperedge
   - Measures feature variance using mean squared distance
   - Converts variance into an attention weight using a Gaussian kernel

3. **Message Modulation**
   - Hyperedge messages are scaled by the variance-based attention weight
   - Low-variance (homogeneous) hyperedges are emphasized
   - High-variance (heterogeneous) hyperedges are suppressed

This design introduces semantic awareness into hyperedge weighting with minimal additional computation.

---

## 🧪 Experimental Results

Experiments were conducted on multiple benchmark datasets:

- **Coauthorship:** Cora, DBLP
- **Cocitation:** Cora, Citeseer, PubMed

Compared to the baseline T-HyperGNN, the proposed VA-T-HyperGNN achieved:

- Consistent improvements across all datasets
- Up to **+19.36% accuracy improvement** on the Citeseer (Cocitation) dataset
- Significant gains on DBLP (+8.34%) and PubMed (+8.00%)
- Only a negligible increase in training time per epoch

These results show that **content-aware hyperedge weighting significantly improves performance**.

---

## 🎯 Contributions

- Identifies limitations of cardinality-based hyperedge weighting in T-HyperGNN
- Proposes variance-based attention to measure semantic coherence within hyperedges
- Improves node classification accuracy with minimal computational overhead
- Demonstrates effectiveness across multiple hypergraph benchmark datasets

---

## 📄 Paper Information

- **Title:** *VA-T-HyperGNN: T-HyperGNN Enhancement via Variance-based Hyperedge Attention*
- **Language:** Written in **Korean**
- **Paper Link:** The full paper can be accessed at the link below  

👉 **Paper:** https://drive.google.com/drive/folders/1a5hwtDlY_KU2EV2XmC2gKNA-y4MxT-cS?usp=sharing

---

## 🧠 One-line Summary

> **An enhanced Hypergraph Neural Network that improves T-HyperGNN by applying variance-based attention to emphasize semantically coherent hyperedges.**
