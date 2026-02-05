# ✈️ Airline Customer Churn Prediction & Segmentation

Machine learning system to predict customer churn and segment customers for targeted retention campaigns.

---

## Project Overview

Analyzes 15,176 airline loyalty program customers to:
1. **Predict churn risk** - Identify customers likely to leave
2. **Segment customers** - Group customers for personalized marketing

---

## Models

### 1. Churn Prediction Model
- **Algorithm:** Random Forest Classifier
- **Performance:** 81% recall, 83% ROC-AUC
- **Output:** Churn risk score (0-100%) and risk category (Low/Medium/High/Critical)
- **Use:** Weekly scoring of all customers for retention campaigns

### 2. Customer Segmentation Model
- **Algorithm:** K-Means Clustering (3 clusters)
- **Segments:**
  - **Road Warriors** (4.3%) - High-value frequent flyers, low churn risk
  - **Loyal Regulars** (65.5%) - Steady customers, very low churn risk
  - **Dabblers** (30.2%) - Low engagement, 24% churn risk ⚠️
- **Use:** Quarterly segmentation for targeted marketing strategies

---

##  Project Structure
```text
├── data/                    # Dataset (see data/README.md for download)
├── notebooks/              # Jupyter notebooks for exploration
│   ├── data_preparation.ipynb
│   ├── kmeans_predict.ipynb
│   ├── rf_predict.ipynb
│   ├── train_kmeans.ipynb
│   └── train_rf.ipynb
├── src/                    # Production Python scripts
│   ├── train_churn_model.py
│   ├── train_clustering_model.py
│   ├── predict_clusters.py
│   └── predict_churn.py
├── models/                 # Trained model artifacts (.pkl)
├── predictions/            # Prediction results
└── docs/ 
    └── architecture.md     #Pipeline architecture
```

---

## Key Features

**21 engineered features including:**
- Behavioral: Total flights, distance, points activity
- Temporal: Month-over-month activity trends
- Derived: Engagement volatility, spending patterns
- Demographics: Age, income, loyalty tier

---

## Results

**Churn Model:**
- Catches 81% of churners before they leave
- Processes 15,000+ customers in 8 minutes
- Identifies 1,302 high-risk customers weekly

**Segmentation:**
- $8.7M revenue at risk in "Dabblers" segment
- 10% improvement = $878K saved annually
- Clear action plans per segment

---

## Architecture

**Deployment:** AWS-native architecture
- **Data Pipeline:** AWS Glue, S3, Step Functions
- **Training:** SageMaker (monthly churn, quarterly clustering)
- **Predictions:** SageMaker Notebook Jobs (weekly batch)
- **Monitoring:** CloudWatch, SageMaker Model Monitor
- **Dashboards:** Amazon QuickSight

**Cost:** ~$200/month infrastructure

---

## Quick Start

1. **Get the data:**
```bash
   # See data/README.md for Kaggle download link
```

2. **Create and activate virtual environment**
```bash
    python3 -m venv .venv
    source .venv/bin/activate
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

4. **Run models:**
```bash
   # Train churn model
   python src/train_churn_model.py
   
   # Train clustering model
   python src/train_clustering_model.py
   
   # Make predictions
   python src/predict.py
```

---

## Known Limitations

1. **1-year data window** - Can't detect seasonal patterns (needs 2-3 years)
2. **Mixed churn definition** - 46% of churned customers never used service
3. **No "why" data** - Can predict WHO will churn but not WHY
4. **Static snapshot** - Can't track customer lifecycle stages over time
5. **Limited features** - Missing customer service data, competitive intel, engagement metrics

---

## Business Metrics

| Metric                    | Value        |
|---------------------------|--------------|
| Customers Scored          | 15,176       |
| High Risk Identified      | 1,302 (8.6%) |
| Revenue at Risk           | $8.7M        |
| Monthly Revenue Protected | $2.4M        |
| Campaign Cost             | $26K/month   |
| ROI                       | 122x         |

---

## Team

**Data Science:** Model development, feature engineering, evaluation  
**Marketing:** Campaign strategy, segment definitions  
**Engineering:** AWS infrastructure, pipeline automation  

---

## Documentation

- **Model Details:** See `notebooks/` for full analysis
- **Architecture:** See `docs/architecture.md` for pipeline architecture

---

##  License

MIT License - Free to use, modify, and distribute.

---

##  Links

- **Kaggle Dataset:** [Link to dataset]
- **Project Presentation:** [Link to slides]
- **Dashboard:** [Link to QuickSight dashboard]