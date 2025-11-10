# ===== employee_salary_app.py =====
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ===== Dataset =====
data = {
    'Experience': [1, 3, 5, 7, 10, 12],
    'Education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master', 'PhD'],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'Finance'],
    'Age': [22, 25, 28, 32, 35, 40],
    'Salary': [30000, 40000, 50000, 65000, 80000, 90000]
}

df = pd.DataFrame(data)

# ===== Encode Categorical Features =====
le_edu = LabelEncoder()
df['Education'] = le_edu.fit_transform(df['Education'])

le_dep = LabelEncoder()
df['Department'] = le_dep.fit_transform(df['Department'])

# ===== Prepare Features =====
X = df[['Experience','Education','Department','Age']].values
y = df['Salary'].values
X_b = np.c_[np.ones((len(X), 1)), X]  # add bias
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # Normal Equation

# ===== Predict Function =====
def predict_salary(exp, edu, dep, age):
    x = np.array([1, exp, edu, dep, age])
    return x.dot(theta)

# ===== Streamlit App =====
st.title("💼 Employee Salary Prediction App")
st.write("Enter employee details to predict salary:")

exp = st.number_input("Experience (Years)", 0, 50, 5)
edu_name = st.selectbox("Education Level", le_edu.classes_)
edu = le_edu.transform([edu_name])[0]

dep_name = st.selectbox("Department", le_dep.classes_)
dep = le_dep.transform([dep_name])[0]

age = st.number_input("Age", 18, 65, 30)

if st.button("Predict Salary"):
    salary = predict_salary(exp, edu, dep, age)
    st.success(f"💰 Estimated Salary: {int(salary):,} $")

# ===== Optional: Graph =====
st.write("### Graph: Experience vs Salary")
plt.scatter(df['Experience'], df['Salary'], color='blue', label='Actual Salary')

exp_range = np.linspace(df['Experience'].min(), df['Experience'].max(), 100)
edu_avg = df['Education'].mean()
dep_avg = df['Department'].mean()
age_avg = df['Age'].mean()

predicted_salaries = [predict_salary(e, edu_avg, dep_avg, age_avg) for e in exp_range]

plt.plot(exp_range, predicted_salaries, color='red', label='Predicted Salary')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary ($)')
plt.legend()
st.pyplot(plt)
