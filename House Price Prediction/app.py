import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal ML Platform", page_icon="🤖", layout="wide")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
st.sidebar.markdown('__Select a mode:__')
page = st.sidebar.radio("Go to:", ["🏋️ Train Model", "🔮 Test Model"])

# Universal paths to save the dynamic dataset model
MODEL_PATH = "models/universal_model.pkl"
SCALER_PATH = "models/universal_scaler.pkl"
COLUMNS_PATH = "models/universal_columns.pkl"
TARGET_PATH = "models/universal_target.pkl"

if page == "🏋️ Train Model":
    st.title("🏋️ Train Your Custom Model")
    st.write("Upload a CSV dataset, choose what column you want to predict, and we will build a custom Machine Learning model tailored entirely to your data!")
    
    uploaded_file = st.file_uploader("Drop your dataset here (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📊 Dataset Preview")
            st.dataframe(df.head())
            
            # Configuration
            st.write("### ⚙️ Configuration")
            target_column = st.selectbox("What exactly are you trying to predict (Target Variable)?", df.columns)
            
            if st.button("🚀 Train Custom Model Now", type="primary"):
                with st.spinner("Analyzing dataset, processing generic features, and training model..."):
                    
                    # 1. Strip Non-Numeric Columns for Robustness
                    # (Standard regressions require numbers. We drop text columns automatically for a generic tool)
                    numeric_df = df.select_dtypes(include=['number'])
                    
                    if target_column not in numeric_df.columns:
                        st.error(f"Error: The target column '{target_column}' contains text! Linear regression requires a numeric target. Please pick a number-based column.")
                        st.stop()
                        
                    # 2. Separate Features and Target
                    X = numeric_df.drop(columns=[target_column])
                    y = numeric_df[target_column]
                    
                    # Save the feature column names so we remember them during Testing
                    feature_columns = list(X.columns)
                    
                    # 3. Handle Missing Values Generically (Fill NA with Median)
                    X = X.fillna(X.median())
                    y = y.fillna(y.median())
                    
                    # 4. Split Data (80% Train, 20% Test)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # 5. Scale features using Standard Scaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 6. Train the Model
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    
                    # 7. Save all dynamic artifacts!
                    joblib.dump(model, MODEL_PATH)
                    joblib.dump(scaler, SCALER_PATH)
                    joblib.dump(feature_columns, COLUMNS_PATH)
                    joblib.dump(target_column, TARGET_PATH)
                    
                    # 8. Evaluate on Test Set
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    st.success("🎉 Custom Model Trained Successfully!")
                    
                    # 9. Display Metrics Beautifully
                    st.write("### Model Performance Report:")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy Score (R²)", f"{r2 * 100:.2f}%", help="How much of the variance in the target your model can explain.")
                    col2.metric("Mean Absolute Error", f"{mae:.2f}", help="Average error when predicting the target.")
                    col3.metric("Root Mean Squared Error", f"{rmse:.2f}", help="Heavily punishes large errors.")
                    
                    st.info("Your model is ready! Go to the **'Test Model'** page on the sidebar to input custom features and predict live.")
                    
        except Exception as e:
            st.error(f"An error occurred reading your file: {e}")

elif page == "🔮 Test Model":
    st.title("🔮 Test Custom Model Predictions")
    
    # Verify a model has been trained first
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(COLUMNS_PATH):
        st.warning("⚠️ No trained custom model found! Please go to the **Train Model** page and build one first.")
    else:
        st.write("Enter specific datapoints below corresponding to your uploaded dataset to test the model.")
        
        # Load the dynamic artifacts
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(COLUMNS_PATH)
        try:
            target_name = joblib.load(TARGET_PATH)
        except Exception:
            target_name = "Output"
            
        st.subheader("Dynamic Feature Inputs")
        
        # We loop through EVERY feature column the model was trained on, and dynamically generate an input box for it!
        user_inputs = {}
        cols = st.columns(3) # Use 3 columns to organize massive datasets cleanly
        
        for index, col_name in enumerate(feature_columns):
            with cols[index % 3]:
                val = st.number_input(f"{col_name}", value=0.0, format="%.2f")
                user_inputs[col_name] = val
                
        st.divider()
        
        # Prediction Engine
        if st.button(f"🔮 Predict {target_name}", type="primary", use_container_width=True):
            
            # Format inputs rigidly matching the training columns structure
            input_df = pd.DataFrame([user_inputs], columns=feature_columns)
            
            # Scale User Inputs
            input_scaled = scaler.transform(input_df)
            
            # Predict
            pred = model.predict(input_scaled)[0]
            
            # Display Big Output
            st.success("✅ Prediction Completed Successfully!")
            st.metric(label=f"Predicted {target_name}", value=f"{pred:,.2f}")
