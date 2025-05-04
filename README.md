# 🚀 FinSight: Investor Risk Analyzer

A dynamic Streamlit-based web app that enables individual investors to assess their risk profiles using real-time stock data or uploaded datasets. Designed as part of **AF3005 – Programming for Finance**, under the supervision of **Dr. Usama Arshad**, this app combines financial insights with machine learning to empower smarter investment decisions.

![FinSight Banner](assets/welcome.gif)

---

## 📊 App Overview

**FinSight** walks users through a complete data analysis pipeline – from data loading to model training and visualization – all within a user-friendly interface. It supports two main data sources: uploaded financial CSVs or real-time stock data from Yahoo Finance.

---

## 🧠 Core Functionalities

✅ **Choose Data Source**  
- Upload your own financial dataset (.csv)  
- Fetch real-time stock data from Yahoo Finance  

✅ **Preprocessing & Feature Engineering**  
- Cleans missing values  
- Generates "Daily Return" and "Volatility" features  

✅ **ML Pipeline**  
- Train/Test Split  
- Logistic Regression model training  
- Feature importance visualization  
- Evaluation with accuracy, confusion matrix, classification report  

✅ **Results Visualization**  
- Interactive scatter plot of risk probabilities  
- Download prediction results as CSV  

✅ **Investor Wisdom Quotes**  
- Random financial wisdom for inspiration  

---

## 🧰 Tech Stack

- **Streamlit** – UI and interactive workflow  
- **yFinance** – Real-time stock data  
- **scikit-learn** – ML pipeline  
- **Plotly** – Interactive visualizations  
- **Pandas / NumPy** – Data processing  

---

## ▶️ Run Locally

```bash
git clone https://github.com/yourusername/finsight-risk-analyzer.git
cd finsight-risk-analyzer
pip install -r requirements.txt
streamlit run app.py
