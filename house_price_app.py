import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===== Dataset =====
data = {
    'Area': [120, 80, 150, 200, 60, 90],
    'Rooms': [3, 2, 4, 5, 2, 3],
    'Age': [10, 20, 5, 2, 30, 15],
    'Price': [900000, 600000, 1200000, 1800000, 500000, 750000]
}
df = pd.DataFrame(data)

# ===== Prepare Features =====
X = df[['Area', 'Rooms', 'Age']].values
y = df['Price'].values
X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# ===== Predict Function =====
def predict_price(area, rooms, age):
    x = np.array([1, area, rooms, age])
    return x.dot(theta)

# ===== Streamlit App =====
st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict price:")

area = st.number_input("Area (m²)", min_value=10, max_value=1000, value=100)
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of building (years)", min_value=0, max_value=100, value=10)

if st.button("Predict Price"):
    price = predict_price(area, rooms, age)
    st.success(f"💰 Estimated Price: {int(price):,} DH")

# ===== Optional: Visual Graph =====
st.write("### Graph: Area vs Price")
plt.scatter(df['Area'], df['Price'], color='blue', label='Actual Price')
area_range = np.linspace(df['Area'].min(), df['Area'].max(), 100)
rooms_avg = df['Rooms'].mean()
age_avg = df['Age'].mean()
predicted_prices = [predict_price(a, rooms_avg, age_avg) for a in area_range]
plt.plot(area_range, predicted_prices, color='red', label='Predicted Price')
plt.xlabel('Area (m²)')
plt.ylabel('Price (DH)')
plt.legend()
st.pyplot(plt)
