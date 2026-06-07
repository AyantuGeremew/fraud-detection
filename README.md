# 🚨 Fraud Detection System – Adey Innovations Inc.
📌 Overview

This project is developed for Adey Innovations Inc., a leading FinTech company serving e-commerce and banking clients. The goal is to build a robust fraud detection system that identifies fraudulent transactions in real time across two different financial domains:

🛒 E-commerce transactions (rich behavioral + user context)
💳 Bank credit card transactions (anonymized PCA features)

Fraud detection is critical for minimizing:

💰 Financial losses (false negatives)
🧑‍💻 Customer frustration (false positives)
🏦 Reputational risk

The system emphasizes balanced classification performance, especially under severe class imbalance.

🎯 Business Objective

Build and evaluate machine learning models that:

Detect fraudulent transactions accurately
Minimize false positives and false negatives
Work effectively on highly imbalanced datasets
Support real-time fraud monitoring and decision-making
Provide interpretable insights using SHAP
📊 Datasets
1. 🛒 E-commerce Fraud Dataset (Fraud_Data.csv)

Contains user-level behavioral and transactional data.

Field	Description
user_id	Unique user identifier
signup_time	User signup timestamp
purchase_time	Transaction timestamp
purchase_value	Transaction amount ($)
device_id	Device identifier
source	Traffic source (SEO, Ads, Direct, etc.)
browser	Browser used
sex	Gender
age	User age
ip_address	IP address used
class	Target (1 = Fraud, 0 = Legitimate)
2. 🌍 IP Mapping Dataset (IpAddress_to_Country.csv)

Used for geolocation enrichment.

Field	Description
lower_bound_ip_address	Start of IP range
upper_bound_ip_address	End of IP range
country	Country mapped to IP range
3. 💳 Credit Card Fraud Dataset (creditcard.csv)

Bank transaction dataset with anonymized features.

Field	Description
Time	Seconds since first transaction
V1–V28	PCA-transformed features
Amount	Transaction amount
Class	Target (1 = Fraud, 0 = Legitimate)
⚙️ Project Workflow
1. Data Preprocessing
Handle missing values
Remove duplicates
Convert timestamps to datetime
Encode categorical variables
Map IP addresses to countries (e-commerce data)
2. Feature Engineering

Key engineered features:

⏱️ time_since_signup
🕒 hour_of_day
📅 day_of_week
⚡ transaction velocity (frequency in time windows)
🔁 transaction frequency per user
🌍 geolocation features (country from IP)
3. Handling Class Imbalance
SMOTE (Synthetic Minority Oversampling Technique)
Class weighting (alternative approach)
Careful separation of train/test data to avoid leakage
4. Model Building

Models used:

Logistic Regression
Random Forest
XGBoost
5. Evaluation Metrics

Due to class imbalance, performance is measured using:

Precision
Recall
F1-score
ROC-AUC
Confusion Matrix

Special focus: Recall for fraud class (minimizing false negatives)

6. Model Explainability
SHAP (SHapley Additive Explanations)
Feature importance analysis
Business interpretation of fraud drivers
📈 Key Insights Expected
High-risk transaction time windows
Behavioral patterns of fraudulent users
Device and browser fraud signals
Geographic fraud hotspots
Early fraud indicators (velocity, signup delay)
🧠 Business Impact

This system enables:

🚨 Real-time fraud detection
💡 Smarter risk-based decision making
📉 Reduced financial losses
😊 Improved customer trust
⚙️ Scalable fraud monitoring across platforms