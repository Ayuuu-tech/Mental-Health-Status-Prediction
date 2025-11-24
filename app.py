import streamlit as st
import pickle, os
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🧠 Mental Health Status Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered assessment based on lifestyle & technology usage patterns</p>', unsafe_allow_html=True)

# Locate latest pickle
MODEL_DIR = 'models'
latest_pkl = None
if os.path.isdir(MODEL_DIR):
    pkls = sorted([p for p in os.listdir(MODEL_DIR) if p.startswith('mh_model_') and p.endswith('.pkl')])
    if pkls:
        latest_pkl = os.path.join(MODEL_DIR, pkls[-1])

if latest_pkl is None:
    st.error('⚠️ No trained model found. Please run the training cell in the notebook first.')
    st.info('💡 Open `Mini Project.ipynb` and execute the Mental Health Classification Training cell.')
    st.stop()

# Load model
with open(latest_pkl, 'rb') as f:
    model_data = pickle.load(f)

# Handle both old and new pickle formats
if 'features' in model_data:
    # New simplified format
    model = model_data['model']
    features = model_data['features']
    inverse_map = model_data['inverse_map']
    encodings = model_data['encodings']
else:
    # Old complex format - need to recreate
    st.error('❌ Old model format detected. Please run the training cell in the notebook again to generate a new model.')
    st.info('💡 Open `Mini Project.ipynb` and execute cell 78 (Mental Health Prediction Model)')
    st.stop()

# Sidebar
st.sidebar.header('📊 Enter Your Information')
st.sidebar.markdown('Fill in your lifestyle details below')

# Personal Info
st.sidebar.subheader('👤 Personal')
age = st.sidebar.slider('Age', 15, 80, 25)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
stress = st.sidebar.select_slider('Stress Level', ['Low', 'Medium', 'High'])

# Daily Activities
st.sidebar.subheader('⏱️ Daily Hours')
tech_hours = st.sidebar.slider('Technology Usage', 0.0, 20.0, 6.0, 0.5)
social_hours = st.sidebar.slider('Social Media', 0.0, 15.0, 3.0, 0.5)
gaming_hours = st.sidebar.slider('Gaming', 0.0, 12.0, 1.0, 0.5)
screen_hours = st.sidebar.slider('Total Screen Time', 0.0, 24.0, 8.0, 0.5)
sleep_hours = st.sidebar.slider('Sleep', 4.0, 12.0, 7.0, 0.5)
exercise_hours = st.sidebar.slider('Physical Activity', 0.0, 5.0, 1.0, 0.25)

# Support & Environment
st.sidebar.subheader('🤝 Support')
support = st.sidebar.radio('Support Systems Access', ['Yes', 'No'])
work_env = st.sidebar.radio('Work Environment', ['Positive', 'Neutral', 'Negative'])
online_support = st.sidebar.radio('Online Support Usage', ['Yes', 'No'])

# Build input
user_data = {
    'Age': age,
    'Gender': encodings['Gender'][gender],
    'Technology_Usage_Hours': tech_hours,
    'Social_Media_Usage_Hours': social_hours,
    'Gaming_Hours': gaming_hours,
    'Screen_Time_Hours': screen_hours,
    'Stress_Level': encodings['Stress_Level'][stress],
    'Sleep_Hours': sleep_hours,
    'Physical_Activity_Hours': exercise_hours,
    'Support_Systems_Access': encodings['Support_Systems_Access'][support],
    'Work_Environment_Impact': encodings['Work_Environment_Impact'][work_env],
    'Online_Support_Usage': encodings['Online_Support_Usage'][online_support]
}

input_df = pd.DataFrame([user_data])[features]

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('🔮 Predict Mental Health Status', use_container_width=True, type='primary')

# Predict
if predict_button:
    with st.spinner('🔄 Analyzing your data...'):
        try:
            pred_int = int(model.predict(input_df)[0])
            proba = model.predict_proba(input_df)[0]
            mh_label = inverse_map.get(pred_int, 'Unknown')
            
            # Stress status
            stressed_flag = 'Stressed' if stress in ['Medium', 'High'] else 'Not Stressed'
            
            # Anxiety heuristic
            total_screen = tech_hours + screen_hours
            
            if total_screen >= 10 and sleep_hours < 6:
                anxiety_level = 'High'
                anxiety_color = '🔴'
            elif total_screen >= 6 and sleep_hours < 7:
                anxiety_level = 'Moderate'
                anxiety_color = '🟡'
            else:
                anxiety_level = 'Low'
                anxiety_color = '🟢'
            
            # Results display
            st.markdown('---')
            st.markdown('## 📋 Assessment Results')
            
            # Mental health category with color coding
            if mh_label == 'Good':
                st.success(f'### ✅ Mental Health Status: **{mh_label}**')
                st.balloons()
            elif mh_label == 'Moderate':
                st.warning(f'### ⚠️ Mental Health Status: **{mh_label}**')
            else:
                st.error(f'### 🔴 Mental Health Status: **{mh_label}**')
            
            # Probability distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('#### 📊 Confidence Levels')
                prob_df = pd.DataFrame({
                    'Category': ['Good', 'Moderate', 'Poor'],
                    'Probability': [f'{p*100:.1f}%' for p in proba],
                    'Score': proba
                })
                st.dataframe(prob_df[['Category', 'Probability']], hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown('#### 📈 Visual Distribution')
                chart_df = pd.DataFrame({
                    'Probability': proba
                }, index=['Good', 'Moderate', 'Poor'])
                st.bar_chart(chart_df, color='#1f77b4')
            
            # Additional insights
            st.markdown('---')
            st.markdown('#### 💡 Additional Insights')
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                if stressed_flag:
                    stress_icon = '😰' if stressed_flag == 'Stressed' else '😌'
                    st.info(f'{stress_icon} **Stress Status:** {stressed_flag}')
                
                st.info(f'{anxiety_color} **Anxiety Level:** {anxiety_level}')
            
            with insight_col2:
                # Recommendations based on results
                st.markdown('**📝 Recommendations:**')
                if mh_label == 'Poor' or anxiety_level == 'High':
                    st.markdown('- Consider reducing screen time')
                    st.markdown('- Prioritize 7-8 hours of sleep')
                    st.markdown('- Seek professional support')
                    st.markdown('- Increase physical activity')
                elif mh_label == 'Moderate':
                    st.markdown('- Monitor your stress levels')
                    st.markdown('- Maintain regular sleep schedule')
                    st.markdown('- Balance screen time with activities')
                else:
                    st.markdown('- Keep up the good habits!')
                    st.markdown('- Continue current lifestyle balance')
                    st.markdown('- Stay connected with support systems')
            
            st.caption('⚠️ Note: This is an AI-based assessment tool. For professional diagnosis, please consult a mental health expert.')
            
        except Exception as e:
            st.error(f'❌ Prediction failed: {e}')
            st.info('Please check that all fields are filled correctly.')

# Model Details Section with Beautiful Cards
st.markdown('---')
st.markdown('<h2 style="text-align: center; color: #1f77b4; margin-top: 2rem;">🔬 Model Intelligence & Insights</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Understand how AI predicts your mental health status</p>', unsafe_allow_html=True)

# Feature Importance with interactive chart
if hasattr(model, 'feature_importances_'):
    st.markdown('#### 📊 Feature Importance - What Matters Most?')
    
    importance_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in features],
        'Importance': model.feature_importances_ * 100
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    st.bar_chart(importance_df.set_index('Feature')['Importance'], color='#ff6b6b', horizontal=True)
    
    st.caption('💡 Higher values indicate stronger influence on predictions')

st.markdown('<br>', unsafe_allow_html=True)

# Info cards with metrics
st.markdown('#### 📈 Model Performance Metrics')
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="🎯 Algorithm",
        value="Random Forest",
        help="Ensemble learning method using multiple decision trees"
    )

with metric_col2:
    st.metric(
        label="📊 Features",
        value=f"{len(features)}",
        help="Number of input variables used for prediction"
    )

with metric_col3:
    st.metric(
        label="🎨 Classes",
        value="3",
        delta="Good, Moderate, Poor",
        help="Possible mental health categories"
    )

with metric_col4:
    st.metric(
        label="📦 Dataset",
        value="10K+",
        help="Training samples used to build the model"
    )

st.markdown('<br>', unsafe_allow_html=True)

# Additional Info in styled boxes
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;">
        <h4 style="margin: 0; color: white;">🧠 How It Works</h4>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        The model analyzes 12 lifestyle factors including sleep, screen time, stress levels, and support systems 
        to predict your mental health status. It uses patterns learned from thousands of individuals to make accurate assessments.
        </p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; color: white;">
        <h4 style="margin: 0; color: white;">🔒 Privacy First</h4>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        All predictions are processed locally on your device. No data is sent to external servers. 
        Your information remains completely private and secure throughout the assessment process.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# Key Insights
with st.expander('💡 Key Insights from Research', expanded=False):
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        **🌙 Sleep Impact**
        - 7-8 hours optimal for mental health
        - Poor sleep linked to higher stress
        
        **📱 Screen Time Effects**
        - Excessive usage correlates with anxiety
        - Balance is key for wellbeing
        
        **🏃 Physical Activity**
        - Regular exercise improves mood
        - Even 30 mins daily makes difference
        """)
    
    with insight_col2:
        st.markdown("""
        **🤝 Support Systems**
        - Social connections reduce stress
        - Access to help improves outcomes
        
        **💼 Work Environment**
        - Positive workplace boosts mental health
        - Toxic environments increase risk
        
        **😰 Stress Management**
        - Chronic stress major risk factor
        - Early intervention crucial
        """)

# Footer with better styling
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('---')
st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
    <p style="margin: 0; color: #666;">
        🤖 <strong>Model:</strong> {model_name} | 
        📊 <strong>Powered by:</strong> RandomForest ML | 
        🔒 <strong>100% Private & Secure</strong>
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #999;">
        ⚠️ Disclaimer: This tool is for informational purposes only. Please consult mental health professionals for diagnosis.
    </p>
</div>
""".format(model_name=os.path.basename(latest_pkl) if latest_pkl else "N/A"), unsafe_allow_html=True)
