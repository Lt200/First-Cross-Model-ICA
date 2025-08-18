# First-Cross-Model-ICA

---

# Cross-Modal Scene Semantic Alignment for Image Complexity Assessment

## Introduction
This repository contains the implementation of our research on **Image Complexity Assessment (ICA)** using multimodal techniques. Unlike traditional CNN-based ICA methods, our approach leverages **generative semantic descriptions** and **contrastive multimodal learning** to achieve state-of-the-art results on the IC9600 dataset.  

<img width="936" height="631" alt="image" src="https://github.com/user-attachments/assets/17dc159f-8d0f-4355-9135-0d544be0f7e9" />



---

## Features
- Semantic description generation via **InstructBLIP**.  
- Modified **CM-SSA** architecture with:  
  - Real Label ↔ MOS Quality  
  - Extended learnable prompts (5 levels)  
  - Upgraded visual encoder (ViT-B/16)  
- Dataset split 8:2 for training/test, supporting large-scale experiments.  

---

## Dataset

We construct our dataset mainly based on IC9600, complemented by VISC-C and Savoias for broader validation:

- IC9600: A high-quality ICA dataset.

  - Images sampled from multiple open-source datasets.

  - Rigorous filtering and outlier removal.

  - Emphasis on consistency, objectivity, and balanced distribution.

- VISC-C: Provides additional subjective complexity ratings, enhancing the diversity of human-annotated references.

- Savoias: A large-scale image aesthetic dataset, leveraged here to test robustness across tasks with complexity–aesthetics correlations.

Together, these datasets enable a comprehensive evaluation across subjective perception, model-generated labels, and cross-domain validation.

---

## Workflow
1. **Text Generation**:  
   - InstructBLIP produces semantic descriptions of images.  

2. **Finetune CM-SSA**: 
   - edit the code(dataloder, train & test code, config set, parameter)

3. **Model Training**:  
   - Dataset modified with Real Label (MOS Quality) and default MOS Align（1）.  
   - Multimodal contrastive learning between visual and textual encoders.  


---
 

### Table 1: Intra-dataset Results

| Model   | IC9600 SRCC↑ | IC9600 PLCC↑ | IC9600 RMSE↓ | VISC-C SRCC↑ | VISC-C PLCC↑ | VISC-C RMSE↓ | Savaois SRCC↑ | Savaois PLCC↑ | Savaois RMSE↓ | Params/M | Flops/G | Speed/ms |
|---------|--------------|--------------|--------------|--------------|--------------|--------------|----------------|----------------|----------------|----------|---------|----------|
| CR      | 0.314        | 0.228        | 0.196        | -            | -            | -            | 0.305          | 0.271          | 0.257          | -        | -       | -        |
| ED      | 0.491        | 0.569        | 0.226        | -            | -            | -            | 0.449          | 0.467          | 0.273          | -        | -       | -        |
| DBCNN   | 0.871        | 0.879        | 0.071        | 0.779        | 0.783        | 0.086        | 0.768          | 0.770          | 0.147          | 15.31    | 86.22   | 23.69    |
| NIMA    | 0.838        | 0.555        | 0.194        | 0.801        | 0.803        | 0.125        | 0.781          | 0.771          | 0.210          | 54.32    | 13.18   | 25.00    |
| HyperIQA| 0.926        | 0.933        | 0.024        | 0.830        | 0.832        | 0.181        | 0.801          | 0.798          | 0.293          | 27.38    | 107.83  | 29.64    |
| CLIPIQ  | 0.897        | 0.898        | 0.078        | 0.781        | 0.796        | 0.122        | 0.779          | 0.794          | 0.101          | -        | 61.07   | 21.76    |
| TOPIQ   | 0.938        | 0.944        | 0.021        | 0.810        | 0.819        | 0.079        | 0.838          | 0.823          | 0.123          | 45.20    | 37.26   | 14.36    |
| CNet    | 0.870        | 0.873        | -            | -            | -            | -            | -              | -              | -              | -        | -       | -        |
| ICNet   | 0.937        | 0.946        | 0.049        | 0.818        | 0.819        | 0.079        | 0.865          | 0.849          | 0.121          | 28.40    | 28.40   | 4.56     |
| **CM-SSA** | **0.958** | **0.961**    | **0.009**    | **0.823**    | **0.805**    | **0.018**    | **0.883**      | **0.875**      | **0.026**      | 205.45   | 52.66   | 35.43    |


---

### Table 2: Cross-dataset Validation

The best performances are highlighted in **bold**.

| Train → Test | IC9600 → VISC-C | IC9600 → Savaois | VISC-C → IC9600 | VISC-C → Savaois | Savaois → IC9600 | Savaois → VISC-C |
|--------------|-----------------|------------------|-----------------|------------------|------------------|------------------|
| Model        | SRCC  | PLCC     | SRCC  | PLCC     | SRCC  | PLCC     | SRCC  | PLCC     | SRCC  | PLCC     | SRCC  | PLCC     |
| DBCNN        | 0.685 | 0.674    | 0.606 | 0.626    | 0.742 | 0.750    | 0.525 | 0.538    | 0.773 | 0.787    | 0.644 | 0.652    |
| NIMA         | 0.566 | 0.297    | 0.603 | 0.240    | 0.714 | 0.390    | 0.575 | 0.540    | 0.818 | 0.725    | 0.630 | 0.588    |
| HyperIQA     | 0.711 | 0.688    | 0.669 | 0.669    | 0.668 | 0.668    | 0.550 | 0.554    | 0.761 | 0.748    | 0.643 | 0.629    |
| CLIPIQ       | 0.702 | 0.686    | 0.600 | 0.608    | 0.406 | 0.399    | 0.487 | 0.481    | 0.789 | 0.775    | 0.550 | 0.568    |
| TOPIQ        | 0.759 | 0.730    | 0.659 | 0.648    | 0.705 | 0.704    | 0.597 | 0.590    | 0.806 | 0.804    | 0.674 | 0.685    |
| ICNet        | 0.753 | 0.712    | **0.728** | **0.716** | **0.718** | **0.739** | **0.664** | **0.660** | 0.805 | 0.807    | **0.712** | **0.709** |
| **CM-SSA**   | **0.760** | **0.740** | 0.710 | 0.712    | 0.577 | 0.556    | 0.541 | 0.486    | **0.838** | **0.828** | 0.702 | 0.681    |




---

## CM-SSA — Main Contribution

### Overview
We introduce CM-SSA, the first cross-modal framework for Image Complexity Assessment (ICA). CM-SSA addresses limitations of existing ICA measures that rely heavily on handcrafted or single-modal deep visual features. By bringing textual scene semantics into the loop, our method produces richer, more generalizable complexity estimates.

### Key ideas

Dual-branch architecture:

Complexity Regression Branch: learns pairwise relationships between images using complexity-level cues to predict fine-grained complexity scores.

Scene Semantic Alignment Branch: aligns visual features with high-level, text-driven scene descriptions to enrich visual cues used for complexity evaluation.

Cross-modal alignment effectively refines visual prompts with semantic context, enabling the model to capture deeper scene structure and meaning beyond low-level cues.

### Results

Outperforms state-of-the-art methods on multiple ICA datasets.

Ablation studies confirm the individual contributions of each branch and demonstrate improved cross-dataset generalization.

Why it matters
CM-SSA moves ICA beyond purely visual feature engineering by integrating semantic scene descriptions, improving robustness and transferability across datasets and scenarios.

---

## Citation
```bash
@article{BMVC2025,
  title={Cross-Modal Scene Semantic Alignment for Image Complexity Assessment},
  author={Y. Luo, Y. Li, J. Liu, J. Fu, H. Amirpour, G. Yue, B. Zhao, P. Corcoran, H. Liu, W. Zhou},
  journal={British Machine Vision Conference},
  year={2025}
}


