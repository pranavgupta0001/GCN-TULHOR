# **GCN-TULHOR: Integrating Graph Convolutional Networks into Trajectory-User Linking with Higher-Order Mobility Flows**

---

## **Overview**

**GCN-TULHOR** is a deep learning framework designed to enhance trajectory-user linking and trajectory understanding by integrating **Graph Convolutional Networks (GCNs)** with transformer-based models. It builds on the existing **Tulhor** architecture by introducing spatial relationships between trajectory points to create a more robust model for tasks such as user classification and trajectory token prediction.

This repository provides an implementation of GCN-TULHOR, including data preparation scripts, model training pipelines, and evaluation utilities.

---

## **Key Features**

- **Graph Convolutional Networks (GCNs):** Enrich trajectory embeddings with spatial relationships using adjacency matrices derived from geospatial data.
- **Transformer Encoder:** Captures sequential dependencies in trajectories through multi-head attention and feed-forward layers.
- **H3 Geospatial Indexing:** Efficiently encodes geographical data and computes spatial neighborhoods.
- **Dual Task Support:**
  - **Masked Language Modeling (MLM):** Predicts masked tokens in trajectories.
  - **User Classification:** Identifies users based on their mobility patterns.
- **Custom Loss Function:** Balances class distribution for imbalanced datasets, improving user classification performance.

---

## **Architecture Overview**

### **1. Data Preparation**
- **Trajectory Tokenization:** Each trajectory is encoded into H3 hexagonal identifiers, representing geographic regions.
- **Graph Construction:** A spatial graph is built with tokens as nodes and edges defined by adjacency in the H3 grid or trajectory sequence.
- **Adjacency Matrix:** Encodes spatial relationships and serves as input to the GCN.

### **2. Model Components**
- **Embedding Layer:** Converts trajectory tokens into dense vector representations.
- **GCN Layer:** Updates embeddings with spatial context using the adjacency matrix.
- **Transformer Encoder:** Processes spatially enriched embeddings to capture sequential dependencies.
- **Output Layers:**
  - **Token Prediction:** Predicts masked trajectory tokens.
  - **User Classification:** Identifies users based on the first token embedding.

---

## **Installation**

### **Requirements**
- Python 3.8+
- PyTorch 1.11+
- Torchtext
- NetworkX
- Scipy
- Numpy
- Pandas
- H3-Py
- TQDM
- Balanced Loss (provided in the repository)

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GCN-TULHOR.git
   cd GCN-TULHOR

   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Test installation:
    ```bash
   python test_setup.py
   ```
# Usage

## Data Preparation
Prepare your dataset with the following columns:
- `user_id`: Unique identifier for the user.
- `higher_order_trajectory`: A space-separated string of H3 indices representing trajectory points.

## Training

### Masked Language Modeling (MLM)
Train the model to predict missing trajectory tokens:
```bash
python train_mlm.py --dataset <path_to_dataset> --epochs 10 --batch_size 128 --learning_rate 1e-4
```

## User Classification
Train the model to classify users based on their trajectories:
```bash
python train_user_classification.py --dataset <path_to_dataset> --epochs 50 --batch_size 64 --learning_rate 2e-4
```

# Evaluation
Evaluate the model on a test dataset:
```bash
python evaluate.py --model <path_to_model> --dataset <path_to_test_data>
```



## **Results**

### **Masked Language Modeling**
- **Task**: Predict missing trajectory tokens based on spatial and sequential context.
- **Performance**: Achieved an accuracy of **X%** on the [dataset name].

### **User Classification**
- **Task**: Predict the user ID associated with a given trajectory.
- **Performance**:
  - **Top-1 Accuracy**: **Y%**
  - **Top-3 Accuracy**: **Z%**
  - **Macro F1-Score**: **W**

---

## **Configuration**

### **Training Parameters**
- `embedding_dim`: Dimension of embeddings (default: 512).
- `hidden_dim`: Hidden size in transformer encoder (default: 512).
- `num_heads`: Number of attention heads (default: 8).
- `gcn_dropout`: Dropout rate in GCN (default: 0.5).
- `epochs`: Number of training epochs (default: 10).
- `batch_size`: Batch size (default: 64).
- `learning_rate`: Learning rate (default: 1e-4).

---

## **Citation**

If you use **GCN-TULHOR** in your research, please cite our work:
```bash
@article{GCN-TULHOR,
  title={GCN-TULHOR: Integrating Graph Convolutional Networks into Trajectory-User Linking with Higher-Order Mobility Flows},
  author={Pranav Gupta and Khoa Tran and Manos Papagelis},
  journal={ACM Journal},
  year={2024},
  volume={X},
  number={Y},
  pages={1--22},
  doi={https://doi.org/XXXXXXX.XXXXXXX}
}
```


## **Contact**

For inquiries or contributions, please reach out to:

- **Pranav Gupta**: [pranavgupta0001@gmail.com](mailto:pranavgupta0001@gmail.com)
- **Khoa Tran**: [khoasimon99@gmail.com](mailto:khoasimon99@gmail.com)
- **Manos Papagelis**: [papaggel@gmail.com](mailto:papaggel@gmail.com)


