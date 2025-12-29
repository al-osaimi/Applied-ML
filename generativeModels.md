# Generative Models

## What is this lecture about?
This lecture explains:
- The difference between **supervised vs unsupervised learning**
- **Discriminative vs generative models**
- Major types of **generative models**
- Why generative models are harder but more powerful

---

## 1️. Supervised vs Unsupervised Learning

### Supervised Learning
- Data: **(x, y)**
- Goal: learn mapping **x → y**
- Examples:
  - Classification
  - Regression
  - Object detection
  - Semantic segmentation
  - Image captioning

> Labels are required.

---

### Unsupervised Learning
- Data: **x only**
- No labels
- Goal: discover **hidden structure** in data

Examples:
- Clustering (K-Means)
- Dimensionality reduction (PCA)
- Feature learning (Autoencoders)
- Density estimation

> No labels are used.

---

## 2️. Discriminative vs Generative Models

Discriminative models learn decision boundaries to **classify data** (e.g., cat vs. dog), focusing on conditional probability. Generative models learn the data's underlying distribution to **create new data** (e.g., generate a new cat image).

### Discriminative Models
- Feature learning (with labels)
- Predict / Assign labels given input
- Examples:
  - Logistic regression
  - SVM
  - CNN classifiers

---

### Generative Models
- Feature learning (without labels)
- Detect outliers
- Generate new samples

---

### Conditional Generative Models
- Generate data conditioned on labels
- Example: generate a *cat image* given label “cat”

---

## 3️. Why Generative Models Are Harder
- Must model the **entire data distribution**
- Require deep understanding of structure
- Much more complex than classification


---

## 4. Taxonomy of Generative Models

### Two Main Categories

#### A) Explicit Density Models
- Explicitly compute **p(x)**
- Can evaluate likelihood
- Examples:
    - Autoregressive models
    - VAEs
    - Normalizing flows


#### B) Implicit Density Models
- Cannot compute **p(x)**
- Can only sample
- Example:
    - GANs

---

## 5. Autoregressive Models

Break probability using chain rule:

$p(x) = \prod_{i} p(x_i | x_1, ..., x_{i-1})$

Each pixel depends on previous pixels.

---

## 7️. PixelRNN

- Generates image **pixel by pixel**
- Uses RNN or LSTM
- Very expressive

> ❌ Very slow (sequential generation)

---

## 8️. PixelCNN

- Still autoregressive
- Uses **CNNs instead of RNNs**
- Faster training (parallel convolutions)
- Generation is still sequential

> Trade-off: speed vs likelihood accuracy.

---

## 9️. Autoencoders (Unsupervised Feature Learning)

### Regular Autoencoder
- Encoder maps input > latent features
- Decoder reconstructs input
- Loss: reconstruction error (L2)

> Learns features without labels but **cannot generate new samples reliably**

---

## 10. Variational Autoencoders (VAE)

### Key Idea
- Probabilistic version of autoencoders
- Learn latent variable $z$
- Can generate new data

---

### Components
- Encoder learns $q(z | x)$
- Decoder learns $p(x | z)$
- Prior over latent space $p(z)$ (usually Gaussian)

---

### Training Objective 
Maximize **ELBO (Evidence Lower Bound)**:
- Reconstruction term
- KL divergence regularization

> Allows sampling and generation.

---

### Why VAEs Are Important

- Combine probabilistic modeling with deep learning
- Can:
  - Generate new samples
  - Learn structured latent space
  - Perform unsupervised learning

---

# Questions

### Q1. What is the difference between discriminative and generative models?
**Answer:**  
Discriminative models learn p(y|x), while generative models learn p(x).

---

### Q2. Why are generative models harder to train?
**Answer:**  
Because they must model the entire data distribution.

---

### Q3. What does an autoencoder learn?
**Answer:**  
Latent features by reconstructing input data without labels.

---

### Q4. Why is PixelRNN slow?
**Answer:**  
Because it generates pixels sequentially.

---

### Q5. What problem does VAE solve compared to regular autoencoders?
**Answer:**  
It enables probabilistic sampling and data generation.

