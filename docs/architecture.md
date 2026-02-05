# ML Pipeline Architecture

## Overview
Automated pipeline for weekly customer churn prediction and quarterly segmentation using AWS services.

---

## Pipeline Flow
```
Data Sources → ETL & Features → Model Training → Predictions → Distribution
```

---

## 1. Data Sources

**Input:**
- Customer loyalty data (flights, points, demographics)
- Stored in Amazon S3 as CSV/Parquet files

**Volume:** ~15,000 customers updated daily

---

## 2. ETL & Feature Engineering

**Process:**
- SageMaker Processing Jobs extract and clean data daily
- Creates 21 features:
  - Behavioral: flights, distance, points activity
  - Temporal: month-over-month trends
  - Derived: engagement volatility, spending patterns

**Output:** Clean feature dataset in S3

---

## 3. Model Training

### Churn Prediction Model
- **Algorithm:** Random Forest Classifier
- **Schedule:** Monthly retraining (1st of month)
- **Performance:** 81% recall, 83% ROC-AUC
- **Output:** Saved model (.pkl) in S3

### Segmentation Model
- **Algorithm:** K-Means Clustering (k=3)
- **Schedule:** Quarterly retraining
- **Segments:** Road Warriors, Loyal Regulars, Dabblers
- **Output:** Saved model + scaler (.pkl) in S3

**Validation:** Models must pass performance gates before deployment

---

## 4. Batch Predictions

**Execution:**
- SageMaker Processing Jobs runs every Monday at 2 AM
- Loads latest models from S3
- Scores all 15,000 customers
- Generates predictions in 8-10 minutes

**Output:**
- Churn risk scores (0-100%)
- Risk categories (Low/Medium/High/Critical)
- Cluster assignments
- Recommended actions

**Storage:** Results saved to S3 as CSV

---

## 5. Distribution

**Predictions flow to:**
- Amazon RDS → Powers dashboards
- Amazon Redshift → Analytics and reporting
- Salesforce (via AWS AppFlow) → CRM integration
- Email notifications → Marketing team

---

## 6. Monitoring

**Automated checks:**
- CloudWatch monitors job execution and errors
- SageMaker Model Monitor detects data/concept drift
- SNS sends alerts for failures or performance degradation

**Dashboards:**
- Amazon QuickSight for business metrics
- CloudWatch for technical monitoring

---

## Technology Stack

| Layer         | Service                    | Purpose                          |
|---------------|----------------------------|----------------------------------|
| Storage       | Amazon S3                  | Data lake, models, predictions   |
| ETL           | SageMaker Processing Jobs  | Data cleaning and transformation |
| Training      | SageMaker Training         | Model development                |
| Predictions   | SageMaker Notebook Jobs    | Weekly batch scoring             |
| Orchestration | EventBridge                | Scheduling                       |
| Monitoring    | CloudWatch + Model Monitor | System and model health          |
| Visualization | QuickSight                 | Business dashboards              |

---

## Schedule

| Task                   | Frequency | Day/Time    |
|------------------------|-----------|-------------|
| Data refresh           | Daily     | 2 AM        |
| Churn predictions      | Weekly    | Monday 2 AM |
| Churn model retraining | Monthly   | 1st at 2 AM |
| Clustering retraining  | Quarterly | 1st at 2 AM |

---

## Cost

**Monthly:** ~$200
- Storage (S3): $10
- Processing (SageMaker): $30
- Training (SageMaker): $100
- Predictions (Notebook Jobs): $20
- Monitoring: $20
- Orchestration (EventBridge): $5

---

## Key Design Decisions

**Batch vs Real-time:** Batch predictions chosen because:
- Retention campaigns are planned (not instant)
- Customer behavior changes slowly
- 99% cheaper than always-on API

**SageMaker Processing Jobs:** Flexible Python-based ETL without Glue complexity

**AWS-native:** Full AWS stack for seamless integration and managed services

---

## Future Enhancements

- Real-time API for website personalization
- Additional features (customer service data, engagement metrics)
- 2-3 years of historical data for better temporal analysis
- A/B testing framework for intervention effectiveness