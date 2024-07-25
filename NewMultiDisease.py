import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Loading the saved models
diabetes_model = pickle.load(open('C:/Users/HP/Desktop/RECORDER/saved models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/HP/Desktop/RECORDER/saved models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/HP/Desktop/RECORDER/saved models/parkinsons_model.sav', 'rb'))
alzheimers_model = pickle.load(open('C:/Users/HP/Desktop/MACHINE LEARNING/ALZHEIMERS PREDICTION/Alzheimer.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           "Alzheimer's Prediction"],
                          icons=['activity', 'heart', 'person', 'activity'],
                          default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
    st.success(parkinsons_diagnosis)


# Alzheimer's Prediction Page
if selected == "Alzheimer's Prediction":
    st.title("Alzheimer's Disease Prediction using ML")

    # Main body input features for Alzheimer's prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age', min_value=0, max_value=120, value=73)
    with col2:
        education = st.selectbox('Education Level', options=[0, 1, 2, 3], index=0)
    with col3:
        bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=22.9)
    with col1:
        sleep_quality = st.slider('Sleep Quality (scale 0-10)', min_value=0.0, max_value=10.0, value=9.0)
    with col2:
        family_history = st.selectbox('Family History of Alzheimer\'s', options=[0, 1], index=0)
    with col3:
        diabetes = st.selectbox('Diabetes', options=[0, 1], index=0)
    with col1:
        depression = st.selectbox('Depression', options=[0, 1], index=0)
    with col2:
        head_injury = st.selectbox('Head Injury', options=[0, 1], index=0)
    with col3:
        hypertension = st.selectbox('Hypertension', options=[0, 1], index=0)
    with col1:
        systolic_bp = st.slider('Systolic BP', min_value=0, max_value=300, value=142)
    with col2:
        diastolic_bp = st.slider('Diastolic BP', min_value=0, max_value=200, value=72)
    with col3:
        chol_total = st.slider('Total Cholesterol (mg/dL)', min_value=0.0, max_value=400.0, value=242.4)
    with col1:
        chol_ldl = st.slider('LDL Cholesterol (mg/dL)', min_value=0.0, max_value=300.0, value=56.2)
    with col2:
        chol_hdl = st.slider('HDL Cholesterol (mg/dL)', min_value=0.0, max_value=150.0, value=33.7)
    with col3:
        chol_trig = st.slider('Triglycerides (mg/dL)', min_value=0.0, max_value=500.0, value=162.2)
    with col1:
        mmse = st.slider('MMSE', min_value=0.0, max_value=30.0, value=21.5)
    with col2:
        functional_assessment = st.slider('Functional Assessment', min_value=0.0, max_value=10.0, value=6.5)
    with col3:
        memory_complaints = st.selectbox('Memory Complaints', options=[0, 1], index=0)
    with col1:
        behavioral_problems = st.selectbox('Behavioral Problems', options=[0, 1], index=0)
    with col2:
        adl = st.slider('ADL', min_value=0.0, max_value=10.0, value=1.7)
    with col3:
        confusion = st.selectbox('Confusion', options=[0, 1], index=0)
    with col1:
        disorientation = st.selectbox('Disorientation', options=[0, 1], index=0)
    with col2:
        personality_changes = st.selectbox('Personality Changes', options=[0, 1], index=0)
    with col3:
        difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', options=[0, 1], index=0)
    with col1:
        forgetfulness = st.selectbox('Forgetfulness', options=[0, 1], index=0)

    # Create a DataFrame with all features
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
            prediction = alzheimers_model.predict(input_data)

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

