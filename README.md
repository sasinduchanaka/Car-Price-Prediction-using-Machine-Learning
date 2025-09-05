# 🚗 Car Price Prediction with Machine Learning + Gemini AI

This project predicts the **selling price of used cars** using machine learning  
and provides **AI explanations** using Google Gemini.

## Features
- Data preprocessing & feature selection
- ML model training (RandomForest)
- Interactive Streamlit web app
- Gemini-powered insights
- INR → USD conversion for predictions
- Prediction history in sidebar


## Project structure
car-price-project/
│
├─ artifacts/                   # Trained models + preprocessors
│   ├─ best_model.joblib
│   ├─ inference_schema.joblib
│
├─ src/
│   ├─ train.py                 # Train ML pipeline + save artifacts
│   ├─ infer_example.py         # Quick test script
│   ├─ app_streamlit.py         # 🚀 Streamlit app
│
├─ data/                        # (optional) raw dataset
│
├─ requirements.txt             # Dependencies
├─ .gitignore                   # Ignored files (artifacts, venv, etc.)
└─ README.md




## Setup

```bash
git clone https://github.com/your-username/car-price-project.git
cd car-price-project
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)
pip install -r requirements.txt
