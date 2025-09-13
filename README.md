# Convolutional Chebyshev VAE (ConvChebyshevVAE)

This repository contains the official PyTorch implementation for the research project "Meta-Learning the Latent Manifold with Learnable-Interaction Neurons." We introduce the ConvChebyshevVAE, a novel non-deterministic unsupervised model designed for high-quality data generation and learning structured, interpretable latent spaces.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Reconstructions from the ConvChebyshevVAE](reconstructions.png)
*<p align="center">High-fidelity reconstructions from the proposed ConvChebyshevVAE (Model E).</p>*

---

## 📖 About The Project

Traditional Variational Autoencoders (VAEs) often produce blurry images and learn entangled latent representations that are difficult to interpret. This project addresses these limitations by introducing a novel VAE architecture that synergistically combines three key innovations:

1.  **Bio-Inspired Chebyshev Layers**: Instead of standard linear layers, our model uses adaptive neurons with input-dependent weights modeled by Chebyshev polynomials. This allows for a more flexible and powerful way to capture complex, non-linear data dependencies.
2.  **Topological Loss**: A regularization term based on Topological Data Analysis (TDA) ensures that the global geometric "shape" of the data manifold is preserved in the latent space.
3.  **Disentanglement Loss**: We employ a loss function that encourages the latent space to factorize into shared, style-invariant features and distinct, augmentation-specific features, leading to a more interpretable representation.

Our proposed model, the **ConvChebyshevVAE**, integrates these components into a deep convolutional backbone. As demonstrated in our research, this model significantly outperforms standard VAEs in generation quality (FID and IS) and reconstruction accuracy on the FashionMNIST dataset.

### Built With

* [PyTorch](https://pytorch.org/)
* [giotto-tda](https://giotto-ai.github.io/gtda-docs/) (for Topological Loss)
* [persim](https://persim.scikit-tda.org/) (for Wasserstein Distance in TDA)
* [UMAP](https://umap-learn.readthedocs.io/en/latest/) (for Latent Space Visualization)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)

---

## 🚀 Getting Started

Follow these instructions to set up the project environment and run the experiments locally.

### Prerequisites

You will need a Python environment manager like Conda. It is highly recommended to use a machine with an NVIDIA GPU and CUDA installed for training.

* **Conda / Miniconda**
    ```sh
    [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
    ```

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/elAbraro/ConvChebyshevVAE.git](https://github.com/elAbraro/ConvChebyshevVAE.git)
    cd ConvChebyshevVAE
    ```

2.  **Create a Conda environment:**
    ```sh
    conda create -n chebyvae python=3.9
    conda activate chebyvae
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is included for easy installation.
    ```sh
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ Usage

The main script to run all experiments is `CSE425_Final_Chebyshev-VAE.py` (or the equivalent Jupyter Notebook). You can select which model to train by modifying the `Args` class within the main execution block.

### Running a Model

To run an experiment, open the Python script or notebook and find the `if __name__ == '__main__':` block at the end. Modify the `model_type` variable to select one of the five architectures:

* `'A'`: **Baseline VAE** (Standard fully-connected VAE)
* `'B'`: **Baseline + Paired Transforms** (Baseline VAE with advanced losses)
* `'C'`: **Chebyshev VAE** (Fully-connected VAE with Chebyshev layers)
* `'D'`: **Chebyshev + Paired Transforms** (Chebyshev VAE with advanced losses)
* `'E'`: **ConvChebyshevVAE** (The proposed convolutional model)

**Example: Running the proposed Model E**

Modify the `Args` class as follows:

```python
if __name__ == '__main__':
    class Args:
        # --- MODEL CONFIGURATION ---
        model_type = 'E' # Selected model
        batch_size = 64
        epochs = 20
        lr = 1e-4
        latent_dim = 20
        cheby_order = 3
        gamma = 0.05       # Weight for topological loss
        delta = 0.5        # Weight for disentanglement loss
        grad_clip = 1.0
        warmup_epochs = 5
        run_name = f'Model_{model_type}_Run' # Output directory name

    args = Args()
    main(args)
