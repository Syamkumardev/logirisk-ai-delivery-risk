# LogiRisk AI – Delivery Risk Intelligence

An ML-powered academic prototype for estimating late-delivery risk in e-commerce logistics.

## Live Application
https://logirisk-ai.streamlit.app/

## Project Overview
LogiRisk AI is an experimental machine learning application that estimates delivery-delay risk using historical order-related features. It provides an interactive interface to explore individual predictions, batch predictions, and simulated order streams.

The application uses a trained XGBoost classification model.

## Key Features
- **Overview:** Project introduction and navigation.
- **Risk Studio:** Analyze an individual order.
- **Batch Analytics:** Upload a CSV, generate predictions, and export results.
- **Live Simulator:** Generate synthetic demo orders and score them automatically.
- **Model Lab:** Explore model-related information.

## Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

## Model Information
**Algorithm:** XGBoost Classifier

**Input features:**
1. Purchase_Year
2. Purchase_Month
3. Purchase_DayOfWeek
4. Purchase_Hour
5. Purchase_Weekend
6. customer_state
7. Seller_State_Count
8. Seller_Cities_Count
9. Item_Count

**Recorded experimental metrics:**
- Average Precision: 0.1543
- ROC-AUC: 0.6187
- Historical late-delivery rate: 8.11%
- Classification threshold: 0.50

These are recorded experimental results, not proof of production readiness.

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/Syamkumardev/logirisk-ai-delivery-risk.git
cd logirisk-ai-delivery-risk
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment (Windows)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 5. Run the application
```bash
python -m streamlit run app.py
```

## Batch Analytics
Download the CSV template from the application and provide the required input columns with the expected data types and valid values.

Upload the CSV, preview the records, generate predictions, and export the results.

## Live Simulator
The simulator generates synthetic example orders and passes them through the existing model.

Start feed, Pause, and Reset demo controls are provided.

**Note:** This is a simulated feed, not a real-time logistics integration.

## Limitations and Responsible Use
- Model scores are not verified or calibrated real-world probabilities.
- Predictions do not guarantee actual delivery outcomes.
- Model performance requires further validation on appropriate unseen data.
- Historical patterns may not represent current logistics conditions.
- The application is an academic research prototype, not a production decision system.

## Future Enhancements
- Validate performance on independent test data.
- Improve model calibration and risk evaluation.
- Integrate authorized real-time logistics data.
- Add model monitoring and drift detection.
- Improve explainability and operational reporting.

## Author
Syam Kumar Appikatla

B.Tech – Computer Science and Engineering (AI & ML)

## Project Status
Deployed academic prototype.
