import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("data/train.csv")

# Preprocess dataset
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'].fillna('S', inplace=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Features and target for model performance
feature_cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']

# --------------------------
# Load trained model
# --------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------
# Sidebar menu
# --------------------------
menu = ["Home", "About", "Data", "Visualisations", "Predict", "Model Performance"]
choice = st.sidebar.selectbox("Menu", menu)

# --------------------------
# Home page
# --------------------------
if choice == "Home":
    st.title("Titanic Survival Prediction App")
    st.write("""
    This app predicts whether a passenger survived the Titanic disaster based on their features.
    Use the sidebar to explore data, visualisations, make predictions, and view model performance.
    """)

# --------------------------
# About page
# --------------------------
elif choice == "About":
    st.title("About This App")
    st.write("""
    **Titanic Survival Prediction App**  
    This project predicts passenger survival on the Titanic using a Random Forest Classifier.  
    It demonstrates an end-to-end Machine Learning pipeline including data preprocessing, model training, evaluation, and deployment via Streamlit.  

    **Developer Contact:**  
    Email: nitharshana1996@gmail.com  
    GitHub Repository: [https://github.com/nitharshana31/ML_STREAMLIT_APP](https://github.com/nitharshana31/ML_STREAMLIT-App)  
    """)

# --------------------------
# Data page
# --------------------------
elif choice == "Data":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

# --------------------------
# Visualisations
# --------------------------
elif choice == "Visualisations":
    st.subheader("Visualisations")

    # Count of survivors
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax1)
    ax1.set_xticklabels(['Did Not Survive', 'Survived'])
    st.pyplot(fig1)

    # Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(df[feature_cols + ['Survived']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Survival by Passenger Class
    fig3, ax3 = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax3)
    ax3.set_xticklabels(['1st Class','2nd Class','3rd Class'])
    st.pyplot(fig3)

# --------------------------
# Prediction
# --------------------------
elif choice == "Predict":
    st.subheader("Enter Passenger Details for Prediction")

    # Input widgets
    Pclass = st.selectbox("Passenger Class (1=1st,2=2nd,3=3rd)", [1,2,3])
    Sex = st.selectbox("Sex", ["male","female"])
    Age = st.number_input("Age", min_value=0, max_value=100, value=30)
    SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    Fare = st.number_input("Fare", 0.0, 600.0, 32.2)
    Embarked = st.selectbox("Port of Embarkation", ["C","Q","S"])

    # Convert inputs
    Sex = 0 if Sex=="male" else 1
    Embarked_Q = 1 if Embarked=="Q" else 0
    Embarked_S = 1 if Embarked=="S" else 0

    if st.button("Predict"):
        input_df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]],
                                columns=feature_cols)
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][prediction]*100

        st.success(f"Predicted Survival: {'Survived' if prediction==1 else 'Did Not Survive'}")
        st.info(f"Prediction Confidence: {prediction_proba:.2f}%")

# --------------------------
# Model Performance
# --------------------------
elif choice == "Model Performance":
    st.subheader("Model Performance Metrics")
    y_true = df['Survived']
    y_pred = model.predict(df[feature_cols])

    st.text("Classification Report")
    st.text(classification_report(y_true, y_pred))

    st.text("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(cm)

    # Heatmap for confusion matrix
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
