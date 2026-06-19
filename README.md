# 🔐 IDS-ML — Network Intrusion Detection System

> Détection d'intrusions réseau par Machine Learning sur le dataset NSL-KDD

![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-green)

## 📌 Description

Ce projet implémente un système de détection d'intrusions réseau (IDS) 
basé sur le Machine Learning. Il analyse le trafic réseau et classifie 
automatiquement les connexions comme **normales** ou **attaques**.

## 🎯 Résultats obtenus

| Métrique | Valeur |
|---|---|
| Connexions analysées | 22 544 |
| Connexions normales détectées | 14 164 |
| Attaques détectées | 8 380 |
| Modèle principal | Random Forest |

## 🛠️ Technologies utilisées

- Python 3.x
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit
- Dataset NSL-KDD

## 📁 Structure du projet
IDS-ML/

├── data/               # Dataset NSL-KDD (non inclus - voir ci-dessous)

├── src/

│   ├── preprocess.py   # Prétraitement des données

│   ├── train.py        # Entraînement des modèles

│   ├── evaluate.py     # Évaluation et comparaison

│   └── app.py          # Interface Streamlit

├── models/             # Modèles sauvegardés

├── requirements.txt

└── README.md
## 🚀 Installation

```bash
git clone https://github.com/VOTRE_USERNAME/IDS-ML.git
cd IDS-ML
pip install -r requirements.txt
```

## 📥 Dataset

Téléchargez le dataset NSL-KDD depuis Kaggle :
👉 https://www.kaggle.com/datasets/hassan06/nslkdd

Placez `KDDTrain+.txt` et `KDDTest+.txt` dans le dossier `data/`.

## ▶️ Utilisation

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
streamlit run src/app.py
```

## 👤 Auteur

**[Nsabiyumva Nice Stella]** — Étudiant en Génie Logiciel, Burundi  
🔗 LinkedIn : [www.linkedin.com/in/nice-nsabiyumva-667660332]

## 📄 Licence

MIT License