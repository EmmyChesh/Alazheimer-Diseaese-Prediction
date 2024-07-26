import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model using pickle
model = pickle.load(open(github.com/EmmyChesh/Alazheimer-Diseaese-Prediction/blob/main/Alzheimerr.sav, 'rb'))

# Initialize the input fields in Streamlit
st.title('Health Data Prediction')
st.write("Provide your health data to predict the likelihood of Alzheimer’s disease.")

# Create a sidebar for navigation and input fields
st.sidebar.title("Input Features")
st.sidebar.write("Fill in the details below:")

# Input fields
age = st.sidebar.slider('Age', min_value=0, max_value=120, value=73)
education = st.sidebar.selectbox('Education Level', options=[0, 1, 2, 3], index=0)
bmi = st.sidebar.slider('BMI', min_value=0.0, max_value=100.0, value=22.9)

# Additional input fields
sleep_quality = st.sidebar.slider('Sleep Quality (scale 0-10)', min_value=0.0, max_value=10.0, value=9.0)

# Additional features
family_history = st.sidebar.selectbox('Family History of Alzheimer\'s', options=[0, 1], index=0)
diabetes = st.sidebar.selectbox('Diabetes', options=[0, 1], index=0)
depression = st.sidebar.selectbox('Depression', options=[0, 1], index=0)
head_injury = st.sidebar.selectbox('Head Injury', options=[0, 1], index=0)
hypertension = st.sidebar.selectbox('Hypertension', options=[0, 1], index=0)
systolic_bp = st.sidebar.slider('Systolic BP', min_value=0, max_value=300, value=142)
diastolic_bp = st.sidebar.slider('Diastolic BP', min_value=0, max_value=200, value=72)
chol_total = st.sidebar.slider('Total Cholesterol (mg/dL)', min_value=0.0, max_value=400.0, value=242.4)
chol_ldl = st.sidebar.slider('LDL Cholesterol (mg/dL)', min_value=0.0, max_value=300.0, value=56.2)
chol_hdl = st.sidebar.slider('HDL Cholesterol (mg/dL)', min_value=0.0, max_value=150.0, value=33.7)
chol_trig = st.sidebar.slider('Triglycerides (mg/dL)', min_value=0.0, max_value=500.0, value=162.2)
mmse = st.sidebar.slider('MMSE', min_value=0.0, max_value=30.0, value=21.5)
functional_assessment = st.sidebar.slider('Functional Assessment', min_value=0.0, max_value=10.0, value=6.5)
memory_complaints = st.sidebar.selectbox('Memory Complaints', options=[0, 1], index=0)
behavioral_problems = st.sidebar.selectbox('Behavioral Problems', options=[0, 1], index=0)
adl = st.sidebar.slider('ADL', min_value=0.0, max_value=10.0, value=1.7)
confusion = st.sidebar.selectbox('Confusion', options=[0, 1], index=0)
disorientation = st.sidebar.selectbox('Disorientation', options=[0, 1], index=0)
personality_changes = st.sidebar.selectbox('Personality Changes', options=[0, 1], index=0)
difficulty_completing_tasks = st.sidebar.selectbox('Difficulty Completing Tasks', options=[0, 1], index=0)
forgetfulness = st.sidebar.selectbox('Forgetfulness', options=[0, 1], index=0)

# Create a DataFrame with all features, without the removed variables
input_data = pd.DataFrame({
    'Age': [age],
    'EducationLevel': [education],
    'BMI': [bmi],
    'SleepQuality': [sleep_quality],
    'FamilyHistoryAlzheimers': [family_history],
    'Diabetes': [diabetes],
    'Depression': [depression],
    'HeadInjury': [head_injury],
    'Hypertension': [hypertension],
    'SystolicBP': [systolic_bp],
    'DiastolicBP': [diastolic_bp],
    'CholesterolTotal': [chol_total],
    'CholesterolLDL': [chol_ldl],
    'CholesterolHDL': [chol_hdl],
    'CholesterolTriglycerides': [chol_trig],
    'MMSE': [mmse],
    'FunctionalAssessment': [functional_assessment],
    'MemoryComplaints': [memory_complaints],
    'BehavioralProblems': [behavioral_problems],
    'ADL': [adl],
    'Confusion': [confusion],
    'Disorientation': [disorientation],
    'PersonalityChanges': [personality_changes],
    'DifficultyCompletingTasks': [difficulty_completing_tasks],
    'Forgetfulness': [forgetfulness]
})

# Predict Button
if st.button('Predict'):
    try:
        # Make the prediction
        prediction = model.predict(input_data)

        # Display the prediction with increased font size
        st.subheader('Prediction:')
        if prediction[0] == 0:
            st.markdown('<p style="font-size:24px; color:green;">Alzheimer Negative</p>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<p style="font-size:24px; color:red;">Alzheimer Positive. See a doctor</p>', unsafe_allow_html=True)

            # Health recommendations based on prediction
            st.subheader("Health Recommendations")
            st.write("""
            - **Consult a Doctor**: Schedule a visit with your healthcare provider to discuss your results.
            - **Healthy Diet**: Adopt a balanced diet with a focus on whole grains, fruits, vegetables, and lean proteins.
            - **Regular Exercise**: Engage in regular physical activity, aiming for at least 30 minutes most days of the week.
            - **Monitor Blood Sugar**: Keep track of your blood sugar levels as recommended by your doctor.
            - **Stay Hydrated**: Drink plenty of water and avoid sugary drinks.
            """)

        # Visualization of input data
        st.subheader("Input Data Visualization")
        input_df = pd.DataFrame([input_data.iloc[0]], columns=input_data.columns)
        st.bar_chart(input_df.T)

    except Exception as e:
        st.write(f"Error: {e}")
