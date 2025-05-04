import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random



# --- Theme Config ---
st.set_page_config(page_title="FinSight: Investor Risk Analyzer", layout="wide")

# --- Helper functions ---
@st.cache_data
def fetch_yahoo_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

# --- Sidebar ---
st.sidebar.title("📊 FinSight Menu")
data_source = st.sidebar.radio("Choose Data Source", ["Upload Kragle Dataset", "Fetch from Yahoo Finance"])

if data_source == "Upload Kragle Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)")
    if ticker:
        df = fetch_yahoo_data(ticker)
        st.sidebar.success(f"Data for {ticker} loaded!")
        
# --- Financial Quote Section ---
quotes = [
    "“In investing, what is comfortable is rarely profitable.” – Robert Arnott",
    "“The stock market is filled with individuals who know the price of everything, but the value of nothing.” – Philip Fisher",
    "“The four most dangerous words in investing are: ‘this time it’s different.’” – Sir John Templeton",
    "“Know what you own, and know why you own it.” – Peter Lynch",
    "“An investment in knowledge pays the best interest.” – Benjamin Franklin",
    "“Risk comes from not knowing what you’re doing.” – Warren Buffett",
    "“The individual investor should act consistently as an investor and not as a speculator.” – Ben Graham",
    "“Investing should be more like watching paint dry or watching grass grow.” – Paul Samuelson"
]
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Investor Wisdom**")
if "quote" not in st.session_state:
    st.session_state.quote = random.choice(quotes)

if st.sidebar.button("🔁 New Quote"):
    st.session_state.quote = random.choice(quotes)

st.sidebar.markdown("---")

st.sidebar.info(st.session_state.quote)


# --- Welcome Message ---
st.markdown("<h1 style='color: gold;'>🚀 Welcome to FinSight</h1>", unsafe_allow_html=True)
st.image("assets/welcome.gif", width=300)
st.markdown("#### Analyze your risk profile and make smarter investment decisions!")

# --- Workflow Buttons ---
if 'step' not in st.session_state:
    st.session_state.step = 0

# Step 1: Load Data
if st.button("1️⃣ Load Data"):
    if 'df' in locals():
        st.dataframe(df.head())
        st.success("✅ Data loaded successfully!")
        st.session_state.step = 1
    else:
        st.warning("⚠️ Please upload or fetch data first.")

# Step 2: Preprocessing
if st.session_state.step >= 1 and st.button("2️⃣ Preprocess Data"):
    df = df.dropna()
    st.info(f"Missing values removed. Shape: {df.shape}")
    st.session_state.step = 2

# Step 3: Feature Engineering
if st.session_state.step >= 2 and st.button("3️⃣ Feature Engineering"):
    if 'Close' in df.columns:
        df['Daily Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].rolling(window=7).std()
        df.dropna(inplace=True)

        # Save to session state
        st.session_state.df = df

        st.success("🚧 Features created: Daily Return, Volatility")
        st.dataframe(df[['Daily Return', 'Volatility']].head())
        st.session_state.step = 3
    else:
        st.error("⚠️ 'Close' column not found in dataset. Ensure you loaded financial data correctly.")


# Step 4: Train/Test Split
if st.session_state.step >= 3 and st.button("4️⃣ Train/Test Split"):
    try:
        df = st.session_state.df  # Fetch updated df with features

        if all(col in df.columns for col in ['Daily Return', 'Volatility']):
            X = df[['Daily Return', 'Volatility']]
            y = (df['Daily Return'] > 0).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Store splits
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.step = 4

            st.success(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
            fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)], title="Data Split")
            st.plotly_chart(fig)

        else:
            st.error("❌ Columns 'Daily Return' and 'Volatility' not found. Run Feature Engineering first.")
    except Exception as e:
        st.error(f"❌ Error during Train/Test split: {e}")



# Step 5: Model Training
if st.session_state.step >= 4 and st.button("5️⃣ Train Model"):
    try:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        model = LogisticRegression()
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.success("✅ Model trained successfully!")
        st.session_state.step = 5
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        # After model training
if st.session_state.step >= 5 and st.button("📌 Show Feature Importance"):
    try:
        model = st.session_state.model
        if hasattr(model, "coef_"):
            coef_df = pd.DataFrame({
                "Feature": st.session_state.X_train.columns,
                "Importance": model.coef_[0]
            })
            fig = px.bar(coef_df, x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(fig)
        else:
            st.warning("Feature importance not available for this model.")
    except Exception as e:
        st.error(f"❌ Couldn't calculate feature importance: {e}")


# Step 6: Evaluation
if st.session_state.step >= 5 and st.button("6️⃣ Evaluate Model"):
    try:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.success(f"🎯 Accuracy: {accuracy:.2f}")
        st.subheader("📊 Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig_cm)

        st.subheader("📄 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        st.session_state.step = 6
    except Exception as e:
        st.error(f"❌ Evaluation failed: {e}")


# Step 7: Results Visualization
if st.session_state.step >= 6 and st.button("7️⃣ Visualize Results"):
    try:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_proba = model.predict_proba(X_test)[:, 1]

        results_df = X_test.copy()
        results_df["Actual"] = y_test.values
        results_df["Probability"] = y_proba

        fig = px.scatter(
            results_df, x="Daily Return", y="Volatility", color="Probability",
            size="Probability", title="Predicted Risk Probabilities"
        )
        st.plotly_chart(fig)

        st.session_state.results_df = results_df
    except Exception as e:
        st.error(f"❌ Visualization failed: {e}")


# After Step 7: Results Visualization
if "results_df" in st.session_state:
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name='prediction_results.csv',
        mime='text/csv',
    )



# --- Final Message ---
st.markdown("---")
st.image("assets/end.gif", width=300)
st.markdown("<h4 style='text-align: center; color: green;'>🎉 Analysis Complete! Save your results and invest wisely.</h4>", unsafe_allow_html=True)