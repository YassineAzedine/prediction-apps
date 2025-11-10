# ===== car_price_app.py =====
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ===== Dataset =====
data = {
    'Year': [2010, 2012, 2015, 2018, 2020, 2016],
    'Brand': ['Toyota', 'BMW', 'Toyota', 'BMW', 'Mercedes', 'Toyota'],
    'Mileage': [120000, 90000, 50000, 30000, 20000, 60000],
    'Fuel': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Petrol'],
    'Transmission': ['Manual', 'Manual', 'Automatic', 'Automatic', 'Automatic', 'Manual'],
    'EngineSize': [1.6, 2.0, 1.8, 2.0, 3.0, 1.6],
    'Price': [5000, 12000, 8000, 15000, 30000, 9000]
}

df = pd.DataFrame(data)

# ===== Encode categorical features =====
le_brand = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])

le_fuel = LabelEncoder()
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])

le_trans = LabelEncoder()
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# ===== Prepare Features =====
X = df[['Year','Brand','Mileage','Fuel','Transmission','EngineSize']].values
y = df['Price'].values
X_b = np.c_[np.ones((len(X), 1)), X]  # add bias
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # Normal Equation

# ===== Predict Function =====
def predict_price(year, brand, mileage, fuel, trans, engine):
    x = np.array([1, year, brand, mileage, fuel, trans, engine])
    return x.dot(theta)

# ===== Streamlit App =====
st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict price:")

year = st.number_input("Year", 1990, 2025, 2015)
brand_name = st.selectbox("Brand", le_brand.classes_)
brand_encoded = le_brand.transform([brand_name])[0]

mileage = st.number_input("Mileage (km)", 0, 500000, 50000)
fuel_name = st.selectbox("Fuel Type", le_fuel.classes_)
fuel_encoded = le_fuel.transform([fuel_name])[0]

trans_name = st.selectbox("Transmission", le_trans.classes_)
trans_encoded = le_trans.transform([trans_name])[0]

engine = st.number_input("Engine Size (L)", 1.0, 6.0, 1.6)

if st.button("Predict Price"):
    price = predict_price(year, brand_encoded, mileage, fuel_encoded, trans_encoded, engine)
    st.success(f"💰 Estimated Car Price: {int(price):,} $")

# ===== Optional: Visual Graph =====
st.write("### Graph: Mileage vs Price")
plt.scatter(df['Mileage'], df['Price'], color='blue', label='Actual Price')

# Predicted Price line using average values for other features
year_avg = df['Year'].mean()
brand_avg = df['Brand'].mean()
fuel_avg = df['Fuel'].mean()
trans_avg = df['Transmission'].mean()
engine_avg = df['EngineSize'].mean()

mileage_range = np.linspace(df['Mileage'].min(), df['Mileage'].max(), 100)
predicted_prices = [predict_price(year_avg, brand_avg, m, fuel_avg, trans_avg, engine_avg) for m in mileage_range]

plt.plot(mileage_range, predicted_prices, color='red', label='Predicted Price')
plt.xlabel('Mileage (km)')
plt.ylabel('Price ($)')
plt.legend()
st.pyplot(plt)
