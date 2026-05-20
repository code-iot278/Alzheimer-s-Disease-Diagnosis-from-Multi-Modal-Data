# ============================================================
# COMPLETE MULTIMODAL ALZHEIMER PIPELINE (ONE CELL)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PHASE 1: MULTIMODAL DATA ACQUISITION (DUMMY DATA)
# ============================================================
print("\nPHASE 1: Multimodal Data Acquisition")

num_patients = 20

# MRI, EEG, EHR, PET dummy features
mri_data = np.random.rand(num_patients, 50)
eeg_data = np.random.rand(num_patients, 30)
ehr_data = np.random.rand(num_patients, 10)
pet_data = np.random.rand(num_patients, 20)

labels = np.random.randint(0, 3, num_patients)   # 3 classes

print("MRI Shape :", mri_data.shape)
print("EEG Shape :", eeg_data.shape)
print("EHR Shape :", ehr_data.shape)
print("PET Shape :", pet_data.shape)

# ============================================================
# PHASE 2: PREPROCESSING
# ============================================================
print("\nPHASE 2: Preprocessing")

scaler = StandardScaler()

mri_pre = scaler.fit_transform(mri_data)
eeg_pre = scaler.fit_transform(eeg_data)
ehr_pre = scaler.fit_transform(ehr_data)
pet_pre = scaler.fit_transform(pet_data)

print("Preprocessing completed")

# ============================================================
# PHASE 3: FEATURE REPRESENTATION LEARNING
# ============================================================
print("\nPHASE 3: Feature Extraction")

# Dummy embedding extraction
mri_features = np.mean(mri_pre, axis=1).reshape(-1,1)
eeg_features = np.mean(eeg_pre, axis=1).reshape(-1,1)
ehr_features = np.mean(ehr_pre, axis=1).reshape(-1,1)
pet_features = np.mean(pet_pre, axis=1).reshape(-1,1)

combined_features = np.concatenate(
    [mri_features, eeg_features, ehr_features, pet_features], axis=1
)

print("Combined Feature Shape:", combined_features.shape)

# ============================================================
# PHASE 4: GRAPH CONSTRUCTION (NGC-BUILDER)
# ============================================================
print("\nPHASE 4: Graph Construction")

similarity = cosine_similarity(combined_features)

threshold = 0.80
G = nx.Graph()

for i in range(num_patients):
    G.add_node(i)

for i in range(num_patients):
    for j in range(i+1, num_patients):
        if similarity[i,j] > threshold:
            G.add_edge(i, j, weight=similarity[i,j])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)
plt.title("Phase 4: NeuroGraph Constructor")
plt.show()

# ============================================================
# PHASE 5: ADAPTIVE PROPAGATION GRAPH ENGINE
# ============================================================
print("\nPHASE 5: APGE-Net")

# Dummy propagation
graph_features = combined_features.copy()

for i in range(num_patients):
    neighbors = list(G.neighbors(i))
    if len(neighbors) > 0:
        graph_features[i] = np.mean(combined_features[neighbors], axis=0)

print("Adaptive propagation completed")

# ============================================================
# PHASE 6: CROSS-MODAL GRAPH FUSION
# ============================================================
print("\nPHASE 6: GraphFusion-X")

fusion_features = (combined_features + graph_features) / 2
print("Fusion Shape:", fusion_features.shape)

# ============================================================
# PHASE 8: CLASSIFICATION (GSP-Net)
# ============================================================
print("\nPHASE 8: Classification")

X_train, X_test, y_train, y_test = train_test_split(
    fusion_features, labels, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ============================================================
# PHASE 10: EVALUATION
# ============================================================
print("\nPHASE 10: Evaluation")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy :", round(acc*100, 2))
print("Precision:", round(prec*100, 2))
print("Recall   :", round(rec*100, 2))
print("F1 Score :", round(f1*100, 2))

# ============================================================
# FINAL OUTPUT TABLE
# ============================================================
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value (%)": [
        acc*100,
        prec*100,
        rec*100,
        f1*100
    ]
})

print("\nFinal Results")
print(results)