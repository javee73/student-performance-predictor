
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write("Analyze student marks and predict final performance.")

# 1. Student Dataset
data = {
    "Name": ["Arun", "Priya", "Kavi", "Rani", "Vijay", "Kumar", "Anu"],
    "Attendance": [90, 75, 95, 80, 85, 70, 92],
    "Assignment": [85, 70, 90, 75, 88, 60, 91],
    "Internal": [80, 65, 92, 78, 84, 58, 89],
    "GPA": [8.5, 7.0, 9.2, 7.8, 8.3, 6.5, 9.0],
    "FinalScore": [88, 72, 95, 82, 86, 65, 93]
}
st.subheader("📂 Upload Student Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
  df = pd.DataFrame(data)


st.subheader("Student Dataset")
st.dataframe(df, use_container_width=True)

# 2. Performance Analysis
marks = np.array(df["FinalScore"])

col1, col2, col3 = st.columns(3)
col1.metric("Average Score", round(np.mean(marks), 2))
col2.metric("Highest Score", int(np.max(marks)))
col3.metric("Lowest Score", int(np.min(marks)))

# 3. Machine Learning Model
X = df[["Attendance", "Assignment", "Internal", "GPA"]]
y = df["FinalScore"]

model = LinearRegression()
model.fit(X, y)

# 4. New Student Input
st.subheader("Predict New Student Score")

with st.form("student_form"):
    attendance = st.number_input(
        "Attendance (%)", 0.0, 100.0, 85.0
    )
    assignment = st.number_input(
        "Assignment Score", 0.0, 100.0, 80.0
    )
    internal = st.number_input(
        "Internal Mark", 0.0, 100.0, 80.0
    )
    gpa = st.number_input(
        "GPA", 0.0, 10.0, 8.0
    )

    submitted = st.form_submit_button("Predict Score")

# 5. Prediction and Recommendation
if submitted:
    new_student = pd.DataFrame({
        "Attendance": [attendance],
        "Assignment": [assignment],
        "Internal": [internal],
        "GPA": [gpa]
    })

    prediction = float(model.predict(new_student)[0])

    st.metric("Predicted Final Score", f"{prediction:.2f}")

    if prediction >= 85:
        st.success("Grade A - Excellent Performance")
    elif prediction >= 70:
        st.info("Grade B - Good Performance, Practice More")
    elif prediction >= 50:
        st.warning("Grade C - Need More Preparation")
    else:
        st.error("Grade D - Focus on Weak Subjects")


# 6. Bar Chart
st.subheader("Student Final Scores")

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(df["Name"], df["FinalScore"])

ax.set_xlabel("Student Name")
ax.set_ylabel("Final Score")
ax.set_ylim(0, 100)

# Student names vertical-ah kaattum
plt.xticks(rotation=90, ha="center")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
