import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

# Set up warning filter
warnings.filterwarnings('ignore')

# Set up page configuration
# st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Header
st.write("<h1 class='header'>*** Personal Fitness Tracker ***</h1>", unsafe_allow_html=True)
st.write("""
    Welcome to the Personal Fitness Tracker! Here, provide your basic details like age, gender, BMI, heart rate, etc., and generate your calories burned.
""")

# Sidebar for User Input
st.sidebar.header("Select your data")

def user_input_features():
    """Allow user to input fitness data via sliders in the sidebar."""
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 22)
    body_temp = st.sidebar.slider("Body Temperature (°C): ", 35.0, 42.0, 37.0)
    duration = st.sidebar.slider("Exercise Duration (minutes): ", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate: ", 50, 180, 80)

    gender = 1 if gender_button == "Male" else 0

    data = {
        "Gender_male": gender,
        "Age": age,
        "BMI": bmi,
        "Body_Temp": body_temp,
        "Duration": duration,
        "Heart_Rate": heart_rate,
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Show user input data
st.write("<h3 class='section-header'>Your Data:</h3>", unsafe_allow_html=True)
st.write(df)

# Show Progress Bar while Loading Data
# st.write("---")
# st.header("Processing Data...")

# Display the progress bar
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Load the dataset and preprocess
calories_data = pd.read_csv("calories.csv")
exercise_data = pd.read_csv("exercise.csv")

# Add BMI column to the exercise data based on Weight and Height
exercise_data["BMI"] = exercise_data["Weight"] / ((exercise_data["Height"] / 100) ** 2)
exercise_data["BMI"] = exercise_data["BMI"].round(2)

# Merge the datasets based on User_ID
exercise_df = exercise_data.merge(calories_data, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=42)

# Prepare the data for model
train_data = train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
test_data = test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# Encoding categorical data
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]

X_test = test_data.drop("Calories", axis=1)
y_test = test_data["Calories"]

# Train the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Align the input data with the training features
df = df.reindex(columns=X_train.columns, fill_value=0)

# Predict the calories burned based on the user input
prediction = model.predict(df)

# Show progress bar for prediction
st.write("---")
st.header("Prediction of Calories Burned :")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"<p class='prediction'>{round(prediction[0], 2)} kcal</p>", unsafe_allow_html=True)

# Show Similar Results Section
st.write("---")
st.header("Similar Results:")

# Find similar results based on predicted calories range
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]

# Display the similar entries as a sample
st.write(similar_data.sample(5))

# General Information Analysis
st.write("---")
st.header("General Information: ")

comparison_age = (exercise_df["Age"] < df["Age"].values[0]).mean() * 100
comparison_duration = (exercise_df["Duration"] < df["Duration"].values[0]).mean() * 100
comparison_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).mean() * 100
comparison_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).mean() * 100

# Show comparison results
st.write("You are older than", round(comparison_age, 2), "% of other people.")
st.write("Your exercise duration is longer than", round(comparison_duration, 2),"% of other people.")
st.write("Your heart rate is higher than", round(comparison_heart_rate, 2), "% of others during exercise.")
st.write("Your body temperature is higher than", round(comparison_body_temp, 2), "% of others during exercise.")

st.write("---")
# st.header("Interactive Visualized Data")

def bmi_calc(bmi_value):
    """Categorize the BMI value into different categories."""
    if bmi_value < 18.5:
        return "Underweight"
    elif 18.5 <= bmi_value < 24.9:
        return "Normal weight"
    elif 25 <= bmi_value < 29.9:
        return "Overweight"
    else:
        return "Obese"


exercise_df["BMI_Category"] = exercise_df["BMI"].apply(bmi_calc)


bmi_counts = exercise_df['BMI_Category'].value_counts().reset_index()
bmi_counts.columns = ['BMI_Category', 'count']  # Rename columns for clarity

st.write("---")
st.write(f"<h3 class='header'>BMI Calculation</h3>", unsafe_allow_html=True)


fig4 = px.bar(
    bmi_counts, 
    x='BMI_Category', 
    y='count', 
    labels={"BMI_Category": "BMI Category", "count": "Number of People"},
    color='BMI_Category', 
    color_discrete_map={
        'Underweight': 'red', 'Normal weight': 'lightgreen', 
        'Overweight': 'lightblue', 'Obese': 'pink'}
)

st.plotly_chart(fig4)


user_bmi_category = bmi_calc(df["BMI"].values[0])
st.write(f"<h3 class='subheader'>Your BMI Result: {user_bmi_category}</h3>", unsafe_allow_html=True)


st.write(f"<h3 class='header'> Duration of Exercise vs Calories Burned</h3>", unsafe_allow_html=True)

fig = px.scatter(exercise_df, x="Duration", y="Calories", color="Gender",
                 labels={"Duration": "Duration in minutes", "Calories": "Calories Burned"})
st.plotly_chart(fig)
