# 📐 The Geometry of Learning: Curvature, Orthogonality, and Computational Cost

<div align="center">

**Notebook:** `CDM__4_Final.ipynb`  
**Datasets:** Breast Cancer Wisconsin, MNIST, Boston Housing  
**Author:** *AmirHossein Talebi Koohestani*  
**University:** *Amir Kabir University of Technology (Tehran Polytechnic)*  
**Supervisor:** Dr. Mehdi Ghatee   
**TA:** Dr. Behnam Yousefimehr

</div>

## 📖 Overview
This project investigates the mathematical and computational foundations of optimization in neural networks, with a focus on the **geometry of loss landscapes**.

Instead of relying solely on standard training loops, the project explores curvature, geometric conditioning, and scalability limits of second-order optimization.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/amirtk09/optimization-geometry-neural-networks.git
cd optimization-geometry-neural-networks
pip install numpy matplotlib scipy scikit-learn tensorflow torch
# or
pip install -r requirements.txt
jupyter notebook
```

Open `CDM__4_Final.ipynb` and run all cells.

## 🎯 Objectives
- Compare GD vs. Newton vs. CG on ill-conditioned functions  
- Benchmark SGD, CG, and L-BFGS on a shallow neural network  
- Analyze Hessian memory growth in deep networks  
- Use QR decomposition to improve conditioning and convergence  

## 🧠 Techniques & Concepts

| Technique | Description |
|----------|-------------|
| Newton & CG | Curvature-aware second‑order optimization |
| Hessian Memory Analysis | Theoretical RAM cost estimation |
| QR Decomposition | Orthogonalizes features, improves conditioning |
| L-BFGS | Low-memory quasi‑Newton method |

## 🧩 Methodology

### 1. Optimization From Scratch
- Implement GD, Newton, and CG  
- Visualize their optimization paths on contour plots  

### 2. Shallow NN Benchmarking
- Train an MLP on the Breast Cancer dataset  
- Compare SGD, L‑BFGS, and SciPy CG  

### 3. Deep Model Scalability
- Train a deep NN on MNIST  
- Compute theoretical Hessian size (GB)  

### 4. Geometric Conditioning (QR)
- Use Boston Housing dataset  
- Orthogonalize data using QR  
- Compare SGD convergence on raw vs. orthogonalized data  

## 📊 Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| Convergence Rate | Iterations/time to reach target loss |
| Hessian Size (GB) | Memory needed for full Hessian |
| Condition Number | Sensitivity of system; reduced via QR |
| Loss Landscape | Visual comparison of optimization paths |
