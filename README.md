# Convolutional Autoencoder and Classifier for Fashion-MNIST with Pre-Training  
### Final Integrative Project – Brunello, Florencia Luciana (2025)

This repository contains the implementation of a deep convolutional autoencoder and a deep convolutional classifier trained on the Fashion-MNIST dataset, including experiments on hyperparameter evaluation, pre-training, fine-tuning, and performance analysis.

The main objective is to study how different architectures, learning rates, dropout levels, and training strategies affect reconstruction quality, feature extraction, and model generalization.

## Repository Structure

- `autoencoder/` – contains the three evaluated autoencoder architectures, including encoder and decoder modules  
- `clasificadora/` – includes the two-layer and three-layer classifier networks  
- `experimentos/` – training scripts (autoencoder, pre-training, fine-tuning, scratch)

## Dataset: Fashion-MNIST

- 60,000 training images  
- 10,000 validation/test images  
- 28 × 28 pixels, grayscale  
- 10 clothing categories  

## Autoencoder Architecture

Three configurations were evaluated by varying the number of filters, kernel sizes, and number of layers.  
**Best configuration:** Experiment 1  
**Best learning rate:** `1e-3`

## Classifier Architecture

The pre-trained encoder is reused as a feature extractor.  
Two types of networks were evaluated: **2-layer** and **3-layer** classifiers.  
**Best architecture:**  
Two-layer classifier with **0.2 dropout**.

## Training Strategies Evaluated

1. **Full training without pre-training**  
2. **Full fine-tuning** using the pre-trained encoder  
3. **Training only the classifier**, keeping the encoder frozen  
4. **Autoencoder pre-training + classifier trained from scratch**

## Main Results

### Autoencoder
- Best validation loss: **0.0025**
- Best learning rate: `1e-3`
- The simplest architecture achieved the best performance

### Classifier
- Deeper architectures show stronger **overfitting**
- Best accuracy:
  - Without pre-training: **92.66%**
  - Fine-tuning: **92.7%** (but with strong overfitting)
- Classifier with frozen encoder:
  - **86.99%**, no overfitting
