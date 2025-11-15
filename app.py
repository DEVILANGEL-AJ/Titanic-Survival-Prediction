import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load the Titanic dataset
data = pd.read_csv('Titanic-Dataset.csv')

# Display basic info about the dataset
st.title("Titanic Survival Prediction App")
st.markdown("""
    This application predicts whether a passenger survived the Titanic disaster 
    based on input features like age, class, and fare.
    """)
st.write("Dataset Information")
st.write(data.info())

# Handle missing data
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Feature engineering: Family size
data['FamilySize'] = data['SibSp'] + data['Parch']

# Define features and target variable
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
X = data[features]
y = data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model accuracy: {accuracy:.2f}")

# Show classification report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'], ax=ax)
plt.title('Confusion Matrix')
fig.savefig("confusion_matrix.png")
st.pyplot(fig)

# Sidebar for user input
st.sidebar.header("Passenger Information")
pclass = st.sidebar.selectbox('Class', [1, 2, 3])
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
fare = st.sidebar.number_input('Fare', min_value=0.0, value=10.0)
embarked = st.sidebar.selectbox('Embarked', ['S', 'C', 'Q'])
family_size = st.sidebar.number_input('Family Size', min_value=0, value=1)

# Map user input to model features
sex = 0 if sex == 'Male' else 1
embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked]

# Prepare input data for prediction
input_data = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]], columns=features)

# Predict survival when the user clicks the button
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.sidebar.write("The passenger survived!")
    else:
        st.sidebar.write("The passenger did not survive.")

# Add interactive pie chart to show survival rate
survival_count = data['Survived'].value_counts()
labels = ['Did Not Survive', 'Survived']
sizes = [survival_count[0], survival_count[1]]
colors = ['#ff9999','#66b3ff']
fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.1, 0))
ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.subheader("Titanic Survival Distribution")
fig_pie.savefig("survival_pie_chart.png")
st.pyplot(fig_pie)
