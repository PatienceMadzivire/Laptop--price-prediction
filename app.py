import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved pipeline and column names
model = joblib.load('best_laptop_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Laptop Price Expert", layout="centered")

st.title("💻 Advanced Laptop Price Predictor")
st.write("This model uses Binary Encoding to handle complex specs like CPU and GPU models.")

# 2. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Brand", ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Others'])
        typename = st.selectbox("Type", ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
        cpu = st.text_input("CPU Model (e.g., Intel Core i7 7500U 2.7GHz)", "Intel Core i5 7200U 2.5GHz")
        ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
        inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6)

    with col2:
        opsys = st.selectbox("Operating System", ['Windows 10', 'No OS', 'Linux', 'macOS', 'Chrome OS', 'Windows 7'])
        gpu = st.text_input("GPU Model (e.g., Intel HD Graphics 620)", "Nvidia GeForce GTX 1050")
        product = st.text_input("Product Name", "MacBook Pro")
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5)
        resolution = st.selectbox("Screen Resolution", ['Full HD 1920x1080', 'IPS Panel Full HD 1920x1080', '1366x768', '4K Ultra HD 3840x2160'])

    submit = st.form_submit_button("Predict Price")

# 3. Prediction Logic
if submit:
    # Create a DataFrame with the exact column names used during training
    input_data = pd.DataFrame([[
        product, company, typename, inches, resolution, cpu, ram, gpu, opsys, weight
    ]], columns=['Product', 'Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Gpu', 'OpSys', 'Weight'])

    # The pipeline handles Binary Encoding for Product/Cpu/Gpu automatically!
    prediction = model.predict(input_data)
    
    st.success(f"### Estimated Price: €{prediction[0]:,.2f}")