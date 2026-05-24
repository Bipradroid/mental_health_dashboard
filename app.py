import streamlit as st 
import plotly.express as px 
import plotly.graph_objects as go 
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
import os
import warnings
import joblib
import pickle
from chatbot import generate_response
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Mental Health Dashboard',page_icon=':dna:',layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    """
    <style>

    [data-testid="collapsedControl"] {
        display: none
    }

    </style>
    """,
    unsafe_allow_html=True
)
@st.cache_data
def load_data():
    return pd.read_csv('mental_health_updated1.csv')
df=load_data()
model = joblib.load("mental_health_model (1).pkl")
with open(
    "feature_columns (1).pkl",
    "rb"
) as f:

    feature_columns = pickle.load(f)

st.markdown("---")
@st.dialog("🧠 Mental Health Assessment")
def prediction_form():

    with st.form("prediction_form"):

        st.markdown("### 👤 Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        country = st.selectbox(
            "Country",
            [
                "australia","belgium",
                "bosnia and herzegovina",
                "brazil","canada","colombia",
                "costa rica","croatia",
                "czech republic","denmark",
                "finland","france","georgia",
                "germany","greece","india",
                "indonesia","ireland",
                "israel","italy","mexico",
                "moldova","netherlands",
                "new zealand","nigeria",
                "philippines","poland",
                "portugal","russia",
                "singapore","south africa",
                "sweden","switzerland",
                "thailand","uk","usa"
            ]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Business",
                "Corporate",
                "Housewife",
                "Others",
                "Student"
            ]
        )

        st.markdown("---")

        st.markdown("### 🧠 Psychological Indicators")

        SelfEmployed = st.selectbox(
            "Self Employed",
            ["No","Yes"]
        )

        FamilyHistory = st.selectbox(
            "Family History",
            ["No","Yes"]
        )

        DaysIndoors = st.selectbox(
            "Days Indoors",
            [
                "Go out Every Day",
                "1-14 days",
                "15-30 days",
                "31-60 days",
                "60+ days"
            ]
        )

        HabitsChange = st.selectbox(
            "Changes In Habits",
            ["No","Maybe","Yes"]
        )

        MentalHealthHistory = st.selectbox(
            "Mental Health History",
            ["No","Maybe","Yes"]
        )

        IncreasingStress = st.selectbox(
            "Increasing Stress",
            ["No","Maybe","Yes"]
        )

        MoodSwings = st.selectbox(
            "Mood Swings",
            ["Low","Medium","High"]
        )

        SocialWeakness = st.selectbox(
            "Social Weakness",
            ["No","Maybe","Yes"]
        )

        CopingStruggles = st.selectbox(
            "Coping Struggles",
            ["No","Yes"]
        )

        WorkInterest= st.selectbox(
            "Work Interest",
            ["No","Maybe","Yes"]
        )

        st.markdown("---")

        st.markdown("### 💬 Mental Health Awareness")

        MentalHealthInterview = st.selectbox(
            "Mental Health Interview",
            ["No","Maybe","Yes"]
        )

        CareOptions = st.selectbox(
            "Care Options",
            ["No","Not Sure","Yes"]
        )

        submitted = st.form_submit_button(
            "🚀 Generate Assessment Report",
            use_container_width=True
        )

        if submitted:
            st.session_state.assessment_data = {

                "Gender": gender,

                "Country": country,

                "Occupation": occupation,

                "SelfEmployed": SelfEmployed,

                "FamilyHistory": FamilyHistory,

                "DaysIndoors": DaysIndoors,

                "HabitsChange": HabitsChange,

                "MentalHealthHistory": MentalHealthHistory,

                "IncreasingStress": IncreasingStress,

                "MoodSwings": MoodSwings,

                "SocialWeakness": SocialWeakness,

                "CopingStruggles": CopingStruggles,

                "WorkInterest": WorkInterest,

                "MentalHealthInterview": MentalHealthInterview,

                "CareOptions": CareOptions
            }
            encoders = {

            "IncreasingStress": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "MoodSwings": {
                "Low": 0,
                "Medium": 1,
                "High": 2
            },

            "HabitsChange": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "SocialWeakness": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "DaysIndoors":{

                "Go out Every Day": 0,

                "1-14 days": 1,

                "15-30 days": 2,

                "31-60 days": 3,

                "60+ days": 4
                },

            "CopingStruggles": {
                "No": 0,
                "Yes": 1
            },

            "FamilyHistory": {
                "No": 0,
                "Yes": 1
            },

            "SelfEmployed": {
                "No": 0,
                "Yes": 1
            },

            "CareOptions": {
                "No": 0,
                "Not Sure": 1,
                "Yes": 2
            },

            "MentalHealthInterview": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "MentalHealthHistory": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "WorkInterest": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
                }
            }
            IncreasingStress = encoders[
                "IncreasingStress"
            ][IncreasingStress]

            MoodSwings = encoders[
                "MoodSwings"
            ][MoodSwings]

            HabitsChange = encoders[
                "HabitsChange"
            ][HabitsChange]

            SocialWeakness = encoders[
                "SocialWeakness"
            ][SocialWeakness]

            DaysIndoors = encoders[
                "DaysIndoors"
            ][DaysIndoors]

            CopingStruggles = encoders[
                "CopingStruggles"
            ][CopingStruggles]

            FamilyHistory = encoders[
                "FamilyHistory"
            ][FamilyHistory]

            SelfEmployed = encoders[
                "SelfEmployed"
            ][SelfEmployed]

            MentalHealthInterview = encoders[
                "MentalHealthInterview"
            ][MentalHealthInterview]

            MentalHealthHistory = encoders[
                "MentalHealthHistory"
            ][MentalHealthHistory]

            WorkInterest = encoders[
                "WorkInterest"
            ][WorkInterest]

            CareOptions = encoders[
                "CareOptions"
            ][CareOptions]
            
            stress_score = (

                0.5 * IncreasingStress

                + CopingStruggles

                + 0.5 * MoodSwings

                + 0.5 * HabitsChange

                + 0.5 * SocialWeakness
                )

            isolation_env = (

                0.25 * DaysIndoors

                + 0.5 * SocialWeakness
                )

            early_stage = int(
                occupation == "Student"
            )
            Gender_Female=int(
                gender=="Female"
            )
            Gender_Male=int(
                gender=="Male"
            )
            
            Occupation_Business = int(
                occupation == "Business"
                )

            Occupation_Corporate = int(
                occupation == "Corporate"
                )

            Occupation_Housewife = int(
                occupation == "Housewife"
            )

            Occupation_Others = int(
                occupation == "Others"
            )

            Occupation_Student = int(
                occupation == "Student"
            )
            
            country_map = {

                "australia":0,
                "belgium":1,
                "bosnia and herzegovina":2,
                "brazil":3,
                "canada":4,
                "colombia":5,
                "costa rica":6,
                "croatia":7,
                "czech republic":8,
                "denmark":9,
                "finland":10,
                "france":11,
                "georgia":12,
                "germany":13,
                "greece":14,
                "india":15,
                "indonesia":16,
                "ireland":17,
                "israel":18,
                "italy":19,
                "mexico":20,
                "moldova":21,
                "netherlands":22,
                "new zealand":23,
                "nigeria":24,
                "philippines":25,
                "poland":26,
                "portugal":27,
                "russia":28,
                "singapore":29,
                "south africa":30,
                "sweden":31,
                "switzerland":32,
                "thailand":33,
                "uk":34,
                "usa":35
            }

            Country_encoded = country_map[
                country.lower()
            ]
            
            X_user = pd.DataFrame([{

                "SelfEmployed": SelfEmployed,

                "FamilyHistory": FamilyHistory,

                "DaysIndoors": DaysIndoors,

                "HabitsChange": HabitsChange,

                "MentalHealthHistory": MentalHealthHistory,

                "IncreasingStress": IncreasingStress,

                "MoodSwings": MoodSwings,

                "SocialWeakness": SocialWeakness,

                "CopingStruggles": CopingStruggles,

                "WorkInterest": WorkInterest,

                "MentalHealthInterview": MentalHealthInterview,

                "CareOptions": CareOptions,

                "Gender_Female": Gender_Female,

                "Gender_Male": Gender_Male,

                "Occupation_Business": Occupation_Business,

                "Occupation_Corporate": Occupation_Corporate,

                "Occupation_Housewife": Occupation_Housewife,

                "Occupation_Others": Occupation_Others,

                "Occupation_Student": Occupation_Student,

                "Country_encoded": Country_encoded,

                "Early stage": early_stage,

                "StressScore": stress_score,

                "Isolation_env": isolation_env

                }])
            X_user = X_user.reindex(
                columns=feature_columns,
                fill_value=0
                )
            
            prediction = model.predict(
                        X_user
                        )[0]

            probability = (
                model
                .predict_proba(X_user)[0]
            )
            confidence = (
                probability.max() * 100
            )
            st.session_state.prediction = prediction

            st.session_state.confidence = confidence

            st.session_state.stress_score = stress_score

            st.session_state.isolation_env = isolation_env
            st.session_state.probability = probability
            st.session_state.early_stage = early_stage


            st.session_state.assessment_context = {

                "Gender": gender,

                "Country": country,

                "Occupation": occupation,

                "FamilyHistory": FamilyHistory,

                "MentalHealthHistory": MentalHealthHistory,

                "IncreasingStress": IncreasingStress,

                "MoodSwings": MoodSwings,

                "SocialWeakness": SocialWeakness,

                "WorkInterest": WorkInterest,

                "StressScore": stress_score,

                "IsolationScore": isolation_env,

                "Prediction": int(prediction),

                "Confidence": round(confidence, 2)
            }

            

            st.session_state.show_report = True


            st.rerun()


@st.dialog("📝 Contribute to Dataset")

def contribution_form():
    with st.form("contribution_form"):
        st.markdown("### 👤 Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        country = st.selectbox(
        "Country",
        sorted(df["Country"].unique())
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Student",
                "Corporate",
                "Business",
                "Housewife",
                "Others"
            ]
        )

        SelfEmployed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        FamilyHistory = st.selectbox(
            "Family History",
            ["No", "Yes"]
        )

        Treatment = st.selectbox(
            "Receiving Treatment",
            ["No", "Yes"]
        )

        st.markdown("---")

        st.markdown("### 🧠 Psychological Indicators")

        DaysIndoors = st.selectbox(
            "Days Indoors",
            [
                "Go out Every Day",
                "1-14 days",
                "15-30 days",
                "31-60 days",
                "60+ days"
            ]
        )

        

        HabitsChange = st.selectbox(
            "Changes in Habits",
            ["No", "Maybe", "Yes"]
        )

        MentalHealthHistory = st.selectbox(
            "Mental Health History",
            ["No", "Maybe", "Yes"]
        )
        IncreasingStress = st.selectbox(
            "Increasing Stress",
            ["No", "Maybe", "Yes"]
        )

        MoodSwings = st.selectbox(
            "Mood Swings",
            ["Low", "Medium", "High"]
        )
        SocialWeakness = st.selectbox(
            "Social Weakness",
            ["No", "Maybe", "Yes"]
        )

        CopingStruggles = st.selectbox(
            "Coping Struggles",
            ["No", "Yes"]
        )

        WorkInterest = st.selectbox(
            "Work Interest",
            ["No", "Maybe", "Yes"]
        )

        

        st.markdown("---")

        st.markdown("### 💬 Mental Health Awareness")

        MentalHealthInterview = st.selectbox(
            "Mental Health Interview",
            ["No", "Maybe", "Yes"]
        )

        CareOptions = st.selectbox(
            "Care Options Available",
            ["No", "Not Sure", "Yes"]
        )

        consent = st.checkbox(
            "I consent to anonymous use of my responses for improving the dataset."
        )

        submitted = st.form_submit_button(
            "🚀 Submit Contribution",
            use_container_width=True
            )
        if submitted and consent:
            encoders = {

            "IncreasingStress": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "MoodSwings": {
                "Low": 0,
                "Medium": 1,
                "High": 2
            },

            "HabitsChange": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "SocialWeakness": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "DaysIndoors":{

                "Go out Every Day": 0,

                "1-14 days": 1,

                "15-30 days": 2,

                "31-60 days": 3,

                "60+ days": 4
                },

            "CopingStruggles": {
                "No": 0,
                "Yes": 1
            },

            "FamilyHistory": {
                "No": 0,
                "Yes": 1
            },

            "Treatment": {
                "No": 0,
                "Yes": 1
            },

            "SelfEmployed": {
                "No": 0,
                "Yes": 1
            },

            "CareOptions": {
                "No": 0,
                "Not Sure": 1,
                "Yes": 2
            },

            "MentalHealthInterview": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "MentalHealthHistory": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
            },

            "WorkInterest": {
                "No": 0,
                "Maybe": 1,
                "Yes": 2
                }
            }
            IncreasingStress = encoders[
                "IncreasingStress"
            ][IncreasingStress]

            MoodSwings = encoders[
                "MoodSwings"
            ][MoodSwings]

            HabitsChange = encoders[
                "HabitsChange"
            ][HabitsChange]

            SocialWeakness = encoders[
                "SocialWeakness"
            ][SocialWeakness]

            DaysIndoors = encoders[
                "DaysIndoors"
            ][DaysIndoors]

            CopingStruggles = encoders[
                "CopingStruggles"
            ][CopingStruggles]

            FamilyHistory = encoders[
                "FamilyHistory"
            ][FamilyHistory]

            Treatment = encoders[
                "Treatment"
            ][Treatment]

            SelfEmployed = encoders[
                "SelfEmployed"
            ][SelfEmployed]

            MentalHealthInterview = encoders[
                "MentalHealthInterview"
            ][MentalHealthInterview]

            MentalHealthHistory = encoders[
                "MentalHealthHistory"
            ][MentalHealthHistory]

            WorkInterest = encoders[
                "WorkInterest"
            ][WorkInterest]

            CareOptions = encoders[
                "CareOptions"
            ][CareOptions]
            stress_score = (

                0.5 * IncreasingStress

                + CopingStruggles

                + 0.5 * MoodSwings

                + 0.5 * HabitsChange

                + 0.5 * SocialWeakness
                )

            isolation_env = (

                0.25 * DaysIndoors

                + 0.5 * SocialWeakness
                )

            early_stage = int(
                occupation == "Student"
            )
            Gender_Female=int(
                gender=="Female"
            )
            new_row = {

            "Gender": gender,

            "Country": country,

            "Occupation": occupation,

            "self_employed": SelfEmployed,

            "family_history": FamilyHistory,

            "Treatment": Treatment,

            "DaysIndoors": DaysIndoors,

            "IncreasingStress": IncreasingStress,

            "Changes_Habits": HabitsChange,

            "Mental_Health_History": MentalHealthHistory,

            "MoodSwings": MoodSwings,

            "Coping_Struggles": CopingStruggles,

            "WorkInterest": WorkInterest,

            "Social_Weakness": SocialWeakness,

            "mental_health_interview": MentalHealthInterview,

            "care_options": CareOptions,

            "StressScore": stress_score,

            "Isolation_env": isolation_env,

            "Early_stage": early_stage
            }
            pd.DataFrame([new_row]).to_csv(
            "user_contributions.csv",
            mode="a",
            header=not os.path.exists("user_contributions.csv"),
            index=False
            )
            st.success(
                    "🎉 Thank you for contributing!"
                )


        

with st.sidebar:

    st.markdown(
    """
    <div style="
        background-color:#161b22;
        border:1px solid #2d333b;
        border-radius:16px;
        padding:35px;
        text-align:left;
        margin-top:10px;
        margin-bottom:10px;
    ">
        <h2 style="color:white;">
            Help us to get Better!
        </h2>

    <p style="
            color:#9ca3af;
            font-size:17px;
            line-height:1.6;
        ">
            We're not perfect!We're constantly working to make our datasets more accurate and inclusive.
            Your anonymous contribution can make a real difference.
            Please fill out this anonymous form to contribute!
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
    if st.button(
        "📝 Contribute Data",
        use_container_width=True
    ):

        contribution_form()
    
    
    st.markdown(
    """
    <div style="
        background-color:#161b22;
        border:1px solid #2d333b;
        border-radius:16px;
        padding:35px;
        text-align:left;
        margin-top:25px;
        margin-bottom:25px;
    ">
        <h2 style="color:white;">
            💙 We are here with You
        </h2>

    <p style="
            color:#9ca3af;
            font-size:17px;
            line-height:1.6;
        ">
            Mental health is personal, and every journey is different.
            Take a quick assessment to understand your current mental
            well-being, compare your results with the community, and
            receive personalized insights powered by machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )



    if st.button(
            "🚀 Start Mental Health Assessment",
            use_container_width=True,
            type="primary"
        ):

            prediction_form()
        
    st.markdown("---")

    st.markdown(
        """
        <div style="
            margin-top:30px;
            text-align:center;
            color:#9ca3af;
            font-size:14px;
        ">
        Need someone to talk to?
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🤖 Talk To Assistant",
        use_container_width=True
    ):
        st.session_state.show_report = False
        st.session_state.open_chat = True

        st.rerun()


st.title(":brain: Mental Health Dashboard")
st.markdown('<style>div.block-container{padding-top:3rem;}</style>',unsafe_allow_html=True)




def semicircle_gauge(value, title):

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",

        value = value,
        number = {
            'suffix': "%"
        },

        title = {'text': title},

        gauge = {
            'axis': {'range': [0, 100]},

            'bar': {'color': "#160B38"},
            'steps': [

                {'range': [0,30], 'color': "#00FF9C"},     # low

                {'range': [30,70], 'color': "#FFD600"},    # moderate

                {'range': [70,100], 'color': "#FF4B4B"}    # high
            ],

            'bgcolor': "#161b22",

            'borderwidth': 2,

            'bordercolor': "#2d333b",

            'shape': "angular"
        }
    ))

    fig.update_layout(
        height=160,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#0e1117",
        font={'color': "white"},
        transition_duration=1000
    )

    return fig
st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #2d333b;
    }

    h1, h2, h3 {
        color: white;
    }

    .css-1d391kg {
        background-color: #111837;
    }

    </style>
    """,
    unsafe_allow_html=True
)

filter1, filter2 = st.columns(2)
country_list = ['All'] + sorted(
    df['Country'].unique().tolist()
)

with filter1:

    selected_country = st.selectbox(
        "🌍 Select Country",
        country_list
    )

with filter2:

    selected_gender = st.selectbox(
        "👤 Select Gender",
        ['All', 'Male', 'Female']
    )
filtered_df=df.copy()
if selected_country != 'All':
    filtered_df = filtered_df[
        filtered_df['Country'] == selected_country
    ]
if selected_gender == 'Male':
    filtered_df = filtered_df[
        filtered_df['Gender_Male'] == 1
    ]





if selected_gender == 'Female':
    filtered_df = filtered_df[
        filtered_df['Gender_Female'] == 1
    ]
col1, col2, col3, col4 = st.columns(4)
total_responses = len(filtered_df)
treatment_percentage = (
    filtered_df['Treatment'].mean() * 100
)

high_stress_percentage = (
    (filtered_df['StressScore'] > filtered_df['StressScore'].mean()).mean() * 100
)

avg_stress_score = filtered_df['StressScore'].mean()
with col1:
    response_percent = (
    len(filtered_df) / len(df)
) * 100

    fig = go.Figure(go.Indicator(

    mode = "gauge+number",

    value = len(filtered_df),

    number = {
        'suffix': ""
    },

    title = {
        'text': "Filtered Responses"
    },

    gauge = {

        'axis': {
            'range': [0,100]
        },

        'bar': {
            'color': "#160B38"
        },
        'steps': [

                {'range': [0,30], 'color': "#00FF9C"},     # low

                {'range': [30,70], 'color': "#FFD600"},    # moderate

                {'range': [70,100], 'color': "#FF4B4B"}    # high
            ],

        'bgcolor': "#161b22",

        'borderwidth': 2,

        'bordercolor': "#2d333b"
    }
    ))

    fig.update_layout(

    height=160,
    margin=dict(l=20, r=20, t=50, b=20),

    paper_bgcolor="#0e1117",

    font={'color':'white'}
    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    config={'displayModeBar': False}
    )
    

with col2:
    fig = semicircle_gauge(
        treatment_percentage,
        "Treatment Required"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    

with col3:
    fig= semicircle_gauge(
        high_stress_percentage,
        "Highly Stressed People"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col4:
    fig = go.Figure(go.Indicator(

    mode = "gauge+number",

    value = avg_stress_score,

    title = {
        'text': "Average Stress Score"
    },

    gauge = {

        'axis': {
            'range': [0,5]
        },

        'bar': {
            'color': "#160B38"
        },
        'steps': [

            {'range': [0,1.2], 'color': "#00FF9C"},     # low

            {'range': [1.2,2.5], 'color': "#FFD600"},    # moderate

            {'range': [2.5,5], 'color': "#FF4B4B"}    # high
        ],

        'bgcolor': "#161b22",

        'borderwidth': 2,

        'bordercolor': "#2d333b"
    }
))

    fig.update_layout(

    height=160,
    margin=dict(l=20, r=20, t=50, b=20),

    paper_bgcolor="#0e1117",

    font={'color':'white'}
    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    config={'displayModeBar': False}
    )

if "show_insights" not in st.session_state:
    st.session_state.show_insights = False
summary_col, btn_col = st.columns([8,1])

with summary_col:

    st.markdown(
        """
        <h1 style="
            color:white;
            font-size:38px;
            margin-bottom:0px;
        ">
        Summery
        </h1>
        """,
        unsafe_allow_html=True
    )

with btn_col:

    if st.button("💡Insights"):
        st.session_state.show_insights = True
    



chart1,chart2,chart3,chart4=st.columns(4)


with chart1:

    with st.container():
        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
            """
            <h4 style="
                color:white;
                text-align:center;
                font-size:15px;
                margin-bottom:0px;
                margin-left:45px;
            ">
            Indoor Time vs Stress
            </h4>
            """,
            unsafe_allow_html=True
            )


        with info_col:

            st.markdown(
        """
            <p title="
            Shows average stress levels for different indoor durations.
            0-> 0 days
            1-> 1-14 days
            2-> 15-30 days
            3-> 31-60 days
            4-> 60+ days
            "
            style="
            color:#9ca3af;
            font-size:15px;
            cursor:pointer;
            text-align:right;
            margin-top:8px;
            ">
            ⓘ
            </p>
            """,
            unsafe_allow_html=True
            )
            
        indoor_stress = filtered_df.groupby(
            'DaysIndoors'
        )['StressScore'].mean().reset_index()

        fig = px.line(
            indoor_stress,
            x='DaysIndoors',
            y='StressScore',
            markers=True
        )

        fig.update_traces(
            line=dict(
                width=2,
                color="#d038fa"
            ),

            marker=dict(
                size=8,
                color="#f414f4"
            )
        )

        fig.update_layout(

            title='',
            height=320,
            width=800,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=10
            ),

            margin=dict(
                l=1,
                r=30,
                t=10,
                b=120
            ),

            xaxis=dict(
                title='Days Indoors',
                showgrid=False,
                zeroline=False
            ),

            yaxis=dict(
                title='Stress Score',
                gridcolor='#2d333b'
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

with chart2:
    with st.container():
        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
            """
            <h4 style="
                color:white;
                text-align:center;
                font-size:15px;
                margin-bottom:0px;
                margin-left:45px;
            ">
            Workholic nature vs Stress
            </h4>
            """,
            unsafe_allow_html=True
            )


        with info_col:

            st.markdown(
            """
                <p title="Workholic behaviour vs stress levels"
                style="
                color:#9ca3af;
                font-size:15px;
                cursor:pointer;
                text-align:right;
                margin-top:4px;
                ">
                ⓘ
                </p>
                """,
        unsafe_allow_html=True
    )
        Work_Interest = filtered_df.groupby(
            'WorkInterest'
        )['StressScore'].mean().reset_index()

        fig = px.line(
            Work_Interest,
            x='WorkInterest',
            y='StressScore',
            markers=True
        )

        fig.update_traces(
            line=dict(
                width=2,
                color="#ff3138"
            ),

            marker=dict(
                size=8,
                color="#de4624"
            )
        )

        fig.update_layout(

            title='',
            height=320,
            width=800,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=10
            ),

            margin=dict(
                l=1,
                r=30,
                t=0,
                b=130
            ),

            xaxis=dict(
                tickmode='array',
                tickvals=[0,1,2],
                title='Work Interest Index',
                showgrid=False,
                zeroline=False
            ),

            yaxis=dict(
                title='Stress Score',
                gridcolor='#2d333b'
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

with chart3:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    text-align: right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Occupation vs Stress
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Average stress score for each occupation"
                style="
                color:#9ca3af;
                font-size:15px;
                cursor:pointer;
                text-align:center;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        occupation_stress = filtered_df.groupby(
            'Occupation'
        )['StressScore'].mean().reset_index()

        fig = px.bar(
            occupation_stress,

            x='Occupation',

            y='StressScore',

            color='StressScore',

            color_continuous_scale='Greens'
        )

        fig.update_layout(

            height=330,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=10,
                b=130
            ),

            xaxis=dict(
                title='Occupation',
                showgrid=False
            ),

            yaxis=dict(
                title='Average Stress Score',

                range=[2.35,2.65],

                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

with chart4:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h5 style="
                    color:white;
                    font-size:15px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Self employment effect
                </h5>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Average stress score among self employed occupations"
                style="
                color:#9ca3af;
                font-size:15px;
                cursor:pointer;
                text-align:right;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        self_emp_df = filtered_df[
            filtered_df['SelfEmployed'] == 1
        ]

        self_emp_stress = self_emp_df.groupby(
            'Occupation'
        )['StressScore'].mean().reset_index()

        fig = px.bar(

            self_emp_stress,

            x='Occupation',

            y='StressScore',

            color='StressScore',

            color_continuous_scale='Blues'
        )

        fig.update_layout(

            height=330,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=10,
                r=20,
                t=10,
                b=130
            ),

            xaxis=dict(
                title='Occupation',
                showgrid=False
            ),

            yaxis=dict(
                title='Average Stress Score',

                range=[2.35,2.65],

                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

top_occ = occupation_stress.loc[
    occupation_stress['StressScore'].idxmax(),
    'Occupation'
]

top_occ_score = occupation_stress[
    'StressScore'
].max()

lowest_indoor = indoor_stress.loc[
    indoor_stress['StressScore'].idxmin(),
    'DaysIndoors'
]  
@st.dialog("💡 Key Insights")

def show_insights_popup():

    st.markdown(f"""

- **{top_occ}** individuals currently show the highest stress score (**{top_occ_score:.2f}**)

- People spending indoor time in no. **{lowest_indoor}** category report the lowest stress levels.

- Stress levels increase significantly with higher work interest.

- Self-employed individuals show consistently elevated stress scores.

    """)
if st.session_state.show_insights:

    show_insights_popup()

    st.session_state.show_insights = False

if "show_dist_insights" not in st.session_state:
    st.session_state.show_dist_insights = False

dist_title, dist_btn = st.columns([8,1])
with dist_title:

    st.markdown(
        """
        <h1 style="
            color:white;
            font-size:38px;
            margin-bottom:-10px;
        ">
        Distributions
        </h1>
        """,
        unsafe_allow_html=True
    )
with dist_btn:

    if st.button("💡Insights", key="dist_insight_btn"):
        st.session_state.show_dist_insights = True


dist1,dist2,dist3=st.columns(3)

with dist1:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    text-align:right;
                    margin-bottom:0px;
                ">
                Occupation Distribution
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Distribution of occupations in the dataset"
                style="
                color:#9ca3af;
                font-size:15px;
                cursor:pointer;
                text-align:right;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        occupation_counts = filtered_df[
            'Occupation'
        ].value_counts().reset_index()

        occupation_counts.columns = [
            'Occupation',
            'Count'
        ]

        fig = px.bar(

            occupation_counts,

            x='Occupation',

            y='Count',

            color='Count',

            text='Count',

            color_continuous_scale='Oranges'
        )

        fig.update_traces(

            textposition='outside'
        )

        fig.update_layout(

            height=300,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',
            uniformtext_minsize=8,
            uniformtext_mode='hide',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=0,
                r=20,
                t=5,
                b=110
            ),

            xaxis=dict(
                title='Occupation',
                tickangle=-20,
                showgrid=False
            ),

            yaxis=dict(
                title='Number of People',
                range=[0, occupation_counts['Count'].max() * 1.15],
                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )    


with dist2:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Stress Score Distribution
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Distribution of stress scores with density curve
                          pink-> Countplot
                          cyan-> Kernel Distribution Plot"
                style="
                color:#9ca3af;
                font-size:15px;
                cursor:pointer;
                text-align:right;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )
        stress_data = filtered_df['StressScore']
        fig = px.histogram(

            filtered_df,

            x='StressScore',

            nbins=30,

            marginal='violin',

            opacity=0.85
        )

        fig.update_traces(

            marker=dict(
                color="#ff24a0"
            )
        )
        kde=gaussian_kde(stress_data)
        x_vals = np.linspace(
                stress_data.min(),
                stress_data.max(),
                200)
        y_vals=kde(x_vals)
        y_vals = y_vals * len(stress_data) * 0.2
        fig.add_trace(

                go.Scatter(

                x=x_vals,

                y=y_vals,

                mode='lines',

                line=dict(
                    color="#00C3FF",
                    width=2
                    ),

        showlegend=False
    )
)

        fig.update_layout(

            height=300,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=0,
                r=20,
                t=5,
                b=110
            ),

            xaxis=dict(
                title='Stress Score',
                showgrid=False
            ),

            yaxis=dict(
                title='Number of People',
                gridcolor='#2d333b'
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )
      

with dist3:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Isolation index Distribution
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Distribution of isolation environment with density curve
                          yellow-> Countplot
                          violet-> Kernel Distribution Plot"
                style="
                color:#9ca3af;
                font-size:15px;
                cursor:pointer;
                text-align:right;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )
        isolation_data = filtered_df['Isolation_env']
        fig = px.histogram(

            filtered_df,

            x='Isolation_env',

            nbins=30,

            marginal='violin',

            opacity=0.85
        )

        fig.update_traces(

            marker=dict(
                color="#f0ff24"
            )
        )
        kde=gaussian_kde(isolation_data)
        x_vals = np.linspace(
                isolation_data.min(),
                isolation_data.max(),
                200)
        y_vals=kde(x_vals)
        y_vals = y_vals * len(isolation_data) * 0.2
        fig.add_trace(

                go.Scatter(

                x=x_vals,

                y=y_vals,

                mode='lines',

                line=dict(
                    color="#961CE3",
                    width=2
                    ),

        showlegend=False
    )
)

        fig.update_layout(

            height=300,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=0,
                r=20,
                t=5,
                b=110
            ),

            xaxis=dict(
                title='Isolation environment index',
                showgrid=False
            ),

            yaxis=dict(
                title='Number of People',
                gridcolor='#2d333b'
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

most_common_occ = occupation_counts.loc[
    occupation_counts['Count'].idxmax(),
    'Occupation'
]

most_common_count = occupation_counts[
    'Count'
].max()

q1 = filtered_df['StressScore'].quantile(0.25)

q3 = filtered_df['StressScore'].quantile(0.75)

iqr = q3 - q1

Q1=filtered_df['Isolation_env'].quantile(0.25)
Q3=filtered_df['Isolation_env'].quantile(0.75)
iqr_iso_env =Q3-Q1


@st.dialog("📊 Distribution Insights")

def show_distribution_popup():

    st.markdown(f"""

- **{most_common_occ}** is currently the most represented occupation
(**{most_common_count:,} people**).

- 50% of individuals report stress scores between
**{q1:.2f}** and **{q3:.2f}**.

- 50% of individuals report isolation environment between
**{Q1:.2f}** and **{Q3:.2f}**.

- Stress scores appear moderately centered with no extreme skewness.

    """)
    
if st.session_state.show_dist_insights:

    show_distribution_popup()

    st.session_state.show_dist_insights = False 
    


st.markdown(
    """
    <h1 style="
        color:white;
        font-size:38px;
        margin-bottom:-10px;
    ">
    Psychological Indicators
    </h1>
    """,
    unsafe_allow_html=True
)

psych1, psych2 = st.columns(2)



with psych1:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Isolation Index for unhealthy individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
        """
        <p title="Isolation environment levels among treatment-seeking individuals"
        style="
        color:#9ca3af;
        font-size:15px;
        text-align:right;
        cursor:pointer;
        margin-top:2px;
        ">
        ⓘ
        </p>
        """,
        unsafe_allow_html=True
    )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 1
        ]
        

        isolation_counts = (
        treatment_df['StressScore']
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .reset_index()
        )

        isolation_counts.columns = [
        'IsolationEnv',
        'Percentage'
        ]

        fig = px.bar(

            isolation_counts,

            x='IsolationEnv',

            y='Percentage',

            color='Percentage',

            text='Percentage',

            color_continuous_scale='Reds'
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            xaxis=dict(
                title='Isolation Environment Index',
                showgrid=False
            ),

            yaxis=dict(
                title='Number of Individuals',
                range=[0, isolation_counts['Percentage'].max() * 1.15],

                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig1',
            config={'displayModeBar': False}
        )


with psych2:

    with st.container():

        title_col, info_col = st.columns([8,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Isolation Index for healthy individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Isolation environment levels among healthy individuals"
                style="
                color:#9ca3af;
                font-size:17px;
                cursor:pointer;
                text-align:right;
                margin-top:0px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 0
        ]
        

        isolation_counts = (
        treatment_df['StressScore']
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .reset_index()
        )

        isolation_counts.columns = [
        'IsolationEnv',
        'Percentage'
        ]


        fig = px.bar(

            isolation_counts,

            x='IsolationEnv',

            y='Percentage',

            color='Percentage',

            text='Percentage',

            color_continuous_scale='Greens'
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            xaxis=dict(
                title='Isolation Environment Index',
                showgrid=False
            ),

            yaxis=dict(
                title='Number of Individuals',
                range=[0, isolation_counts['Percentage'].max() * 1.15],

                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig2',
            config={'displayModeBar': False}
        )

psych3, psych4 = st.columns(2)

with psych3:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Stress Score Distribution Percentages
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Stress score distribution among treatment-seeking individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 1
        ]

        stress_counts = (
        treatment_df['StressScore']
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .reset_index()
        )

        stress_counts.columns = [
        'StressScore',
        'Percentage'
        ]

        fig = px.bar(

            stress_counts,

            x='StressScore',

            y='Percentage',

            color='Percentage',

            text='Percentage',

            color_continuous_scale='Reds'
        )

        fig.update_traces(
            

            texttemplate='%{text:.2f}%',

            textposition='outside'
            )
               
        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',
            

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            xaxis=dict(
                title='Stress Score',
                showgrid=False
            ),

            yaxis=dict(
                title='Number of Individuals',
                range=[0,stress_counts['Percentage'].max() * 1.15],
                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig3',
            config={'displayModeBar': False}
        )
        
with psych4:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Stress Score Distribution Percentages
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Stress score distribution among healthy individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 0
        ]

        stress_counts = (
            treatment_df['StressScore']
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .reset_index()
            )

        stress_counts.columns = [
        'StressScore',
        'Percentage'
        ]

        fig = px.bar(

            stress_counts,

            x='StressScore',

            y='Percentage',

            color='Percentage',

            text='Percentage',

            color_continuous_scale='Greens'
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            xaxis=dict(
                title='Stress Score',
                showgrid=False
            ),

            yaxis=dict(
                title='Number of Individuals',
                range=[0,stress_counts['Percentage'].max() * 1.15],
                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig4',
            config={'displayModeBar': False}
        )

       
psych5,psych6=st.columns(2)

with psych5:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:center;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Occurence of Family History for unhealthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Presence of Family History among treatment-seeking individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 1
        ]

        FamilyHistory_counts = (
            treatment_df['FamilyHistory']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        FamilyHistory_counts.columns = [
            'FamilyHistory',
            'Percentage'
        ]
        FamilyHistory_counts['FamilyHistory'] = (
        FamilyHistory_counts['FamilyHistory']
        .map({
            0: 'No',
            1: 'Yes'
            })
        )

        fig = px.pie(

            FamilyHistory_counts,

            names='FamilyHistory',

            values='Percentage',

            color_discrete_sequence=[
                
                "#c51717",
                "#10921f"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=15,

            hole=0.58,

            pull=[0.03, 0.03],
                marker=dict(

                    line=dict(
                    color="#ccbcbc",
                    width=3
                    )
                )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            annotations=[

                dict(

                    
                    text='Family History',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ],

            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig5',
            config={'displayModeBar': False}
        )
 
with psych6:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:center;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Occurence of Family History for healthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Presence of Family History among healthy individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 0
        ]

        FamilyHistory_counts = (
            treatment_df['FamilyHistory']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        FamilyHistory_counts.columns = [
            'FamilyHistory',
            'Percentage'
        ]
        FamilyHistory_counts['FamilyHistory'] = (
        FamilyHistory_counts['FamilyHistory']
        .map({
            0: 'No',
            1: 'Yes'
            })
        )

        fig = px.pie(

            FamilyHistory_counts,

            names='FamilyHistory',

            values='Percentage',

            color_discrete_sequence=[
                "#10921f",
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=15,

            hole=0.58,

            pull=[0.03, 0.03],
                marker=dict(

                    line=dict(
                    color="#ccbcbc",
                    width=3
                    )
                )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            annotations=[

                dict(

                    
                    text='Family History',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ],

            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig6',
            config={'displayModeBar': False}
        )
 


psych7, psych8 = st.columns(2)

with psych7:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Coping Struggles for unhealthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Coping struggle levels among treatment-seeking individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 1
        ]

        coping_counts = (
            treatment_df['CopingStruggles']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        coping_counts.columns = [
            'CopingStruggles',
            'Percentage'
        ]
        coping_counts['CopingStruggles'] = (
        coping_counts['CopingStruggles']
        .map({
            0: 'No',
            1: 'Yes'
            })
        )

        fig = px.pie(

            coping_counts,

            names='CopingStruggles',

            values='Percentage',

            color_discrete_sequence=[
                "#10921f",
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=15,

            hole=0.58,

            pull=[0.03, 0.03],
                marker=dict(

                    line=dict(
                    color="#ccbcbc",
                    width=3
                    )
                )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            annotations=[

                dict(

                    
                    text='Coping Struggles',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ],

            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig7',
            config={'displayModeBar': False}
        )
        
with psych8:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Coping Struggles for healthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Coping struggle levels among healthy individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 0
        ]

        coping_counts = (
            treatment_df['CopingStruggles']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        coping_counts.columns = [
            'CopingStruggles',
            'Percentage'
        ]
        coping_counts['CopingStruggles'] = (
        coping_counts['CopingStruggles']
        .map({
            0: 'No',
            1: 'Yes'
            })
        )

        fig = px.pie(

            coping_counts,

            names='CopingStruggles',

            values='Percentage',

            color_discrete_sequence=[
                "#10921f",
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=15,

            hole=0.58,

            pull=[0.03, 0.03],
                marker=dict(

                    line=dict(
                    color="#ccbcbc",
                    width=3
                    )
                )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            annotations=[

                dict(

                    
                    text='Coping Struggles',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ],

            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig8',
            config={'displayModeBar': False}
        )
 
psych9,psych10=st.columns(2)

with psych9:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Social Weakness for unhealthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Social weakness indicators among treatment-seeking individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 1
        ]

        social_counts = (
            treatment_df['SocialWeakness']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        social_counts.columns = [
            'SocialWeakness',
            'Percentage'
        ]

        social_counts['SocialWeakness'] = (
            social_counts['SocialWeakness']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
                
            })
        )

        fig = px.pie(

            social_counts,

            names='SocialWeakness',

            values='Percentage',

            color_discrete_sequence=[
                "#e98e27",
                '#10921f',
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=14,

            hole=0.6,

            pull=[0.03, 0.03,0.03],

            marker=dict(

                line=dict(
                    color="#ccbcbc",
                    width=3
                )
            )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            showlegend=False,

            uniformtext_minsize=12,

            uniformtext_mode='hide',

            annotations=[

                dict(

                    
                    text='Social',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig9',
            config={'displayModeBar': False}
        )


with psych10:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Social Weakness for healthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Social weakness indicators among healthy individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 0
        ]

        social_counts = (
            treatment_df['SocialWeakness']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        social_counts.columns = [
            'SocialWeakness',
            'Percentage'
        ]

        social_counts['SocialWeakness'] = (
            social_counts['SocialWeakness']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
                
            })
        )

        fig = px.pie(

            social_counts,

            names='SocialWeakness',

            values='Percentage',

            color_discrete_sequence=[
                "#e98e27",
                '#10921f',
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=14,

            hole=0.6,

            pull=[0.03, 0.03,0.03],

            marker=dict(

                line=dict(
                    color="#ccbcbc",
                    width=3
                )
            )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            showlegend=False,

            uniformtext_minsize=12,

            uniformtext_mode='hide',

            annotations=[

                dict(

                    
                    text='Social',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig10',
            config={'displayModeBar': False}
        )

psych11,psych12=st.columns(2)

with psych11:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Mood Swings for unhealthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Moodswing indicators among treatment-seeking individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 1
        ]

        Moodswings_counts = (
            treatment_df['MoodSwings']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        Moodswings_counts.columns = [
            'MoodSwings',
            'Percentage'
        ]

        Moodswings_counts['MoodSwings'] = (
            Moodswings_counts['MoodSwings']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
                
            })
        )

        fig = px.pie(

            Moodswings_counts,

            names='MoodSwings',

            values='Percentage',

            color_discrete_sequence=[
                "#e98e27",
                '#10921f',
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=14,

            hole=0.6,

            pull=[0.03, 0.03,0.03],

            marker=dict(

                line=dict(
                    color="#ccbcbc",
                    width=3
                )
            )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            showlegend=False,

            uniformtext_minsize=12,

            uniformtext_mode='hide',

            annotations=[

                dict(

                    
                    text='Mood Swings',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig11',
            config={'displayModeBar': False}
        )

with psych12:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:17px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Mood Swings for healthy Individuals
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Moodswing indicators among healthy individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        treatment_df = filtered_df[
            filtered_df['Treatment'] == 0
        ]

        Moodswings_counts = (
            treatment_df['MoodSwings']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        Moodswings_counts.columns = [
            'MoodSwings',
            'Percentage'
        ]

        Moodswings_counts['MoodSwings'] = (
            Moodswings_counts['MoodSwings']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
                
            })
        )

        fig = px.pie(

            Moodswings_counts,

            names='MoodSwings',

            values='Percentage',

            color_discrete_sequence=[
                "#e98e27",
                '#10921f',
                "#c51717"
            ]
        )

        fig.update_traces(

            textposition='outside',

            textinfo='percent+label',

            textfont_size=14,

            hole=0.6,

            pull=[0.03, 0.03,0.03],

            marker=dict(

                line=dict(
                    color="#ccbcbc",
                    width=3
                )
            )
        )

        fig.update_layout(

            height=320,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            showlegend=False,

            uniformtext_minsize=12,

            uniformtext_mode='hide',

            annotations=[

                dict(

                    
                    text='Mood Swings',
                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key='fig12',
            config={'displayModeBar': False}
        )
        
    

st.markdown(
    """
    <h1 style="
        color:white;
        font-size:38px;
        margin-bottom:-10px;
    ">
    Early Stage Indicators
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h3 style="
        color:white;
        font-size:20px;
        margin-bottom:10px;
    ">
    Primary Indicators:
    </h3>
    """,
    unsafe_allow_html=True
)

gr1,gr2,gr3,gr4,gr5=st.columns(5)

with gr1:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    text-align:right;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Early Stage Reports
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Percentage of individuals identified in early mental health stages"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_percentage = (
            filtered_df['Early stage']
            .mean()
        ) * 100

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=early_stage_percentage,

            number={
                'suffix': "%"
            },

            title={
                'text': ""
            },

            gauge={

                'axis': {
                    'range': [0, 100]
                },

                'bar': {
                    'color': "#000000"
                },

                'bgcolor': "#161b22",

                'borderwidth': 2,

                'bordercolor': "#2d333b",

                'steps': [

                    {
                        'range': [0, 30],
                        'color': "#22c55e"
                    },

                    {
                        'range': [30, 60],
                        'color': "#facc15"
                    },

                    {
                        'range': [60, 100],
                        'color': "#ef4444"
                    }
                ]
            }
        ))

        fig.update_layout(

            height=260,
            # width=500,

            paper_bgcolor="#0e1117",

            font={
                'color': 'white'
            },

            margin=dict(
                l=20,
                r=30,
                t=50,
                b=20
            )
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key="early_stage_gauge"
        )

with gr2:


    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Family History in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Family history distribution among individuals identified in early mental health stages"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        family_counts = (
            early_stage_df['FamilyHistory']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        family_counts.columns = [
            'FamilyHistory',
            'Percentage'
        ]

        family_counts['FamilyHistory'] = (
            family_counts['FamilyHistory']
            .map({
                0: 'No',
                1: 'Yes'
            })
        )

        fig = px.bar(

            family_counts,

            x='FamilyHistory',

            y='Percentage',

            color='Percentage',

            text='Percentage',

            color_continuous_scale='Purples'
        )

        fig.update_traces(

            texttemplate='%{text:.2f}%',

            textposition='outside'
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            xaxis=dict(
                title='Family History',
                showgrid=False
            ),

            yaxis=dict(
                title='Percentage of Individuals',

                range=[
                    0,
                    family_counts['Percentage'].max() * 1.15
                ],

                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='family_history_early_stage'
        )
        
with gr3:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Coping Struggles in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Coping struggle distribution among early-stage individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        coping_counts = (
            early_stage_df['CopingStruggles']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        coping_counts.columns = [
            'CopingStruggles',
            'Percentage'
        ]

        coping_counts['CopingStruggles'] = (
            coping_counts['CopingStruggles']
            .map({
                0: 'No',
                1: 'Yes'
            })
        )

        fig = px.pie(

            coping_counts,

            names='CopingStruggles',

            values='Percentage',

            color_discrete_sequence=[
                "#f62424",
                "#3fe925"
            ]
        )

        fig.update_traces(


            textfont_size=14,
            

            hole=0.5,

            pull=[0.025, 0.025],

            marker=dict(

                line=dict(
                    color='#0e1117',
                    width=4
                )
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=35,
                r=0,
                t=30,
                b=50
            ),

            showlegend=False,

            annotations=[

                dict(

                    text='Coping',

                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='coping_early_stage'
        )

with gr4:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Mental Health Disclosure Trends
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Willingness of early-stage individuals to discuss mental health openly"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        interview_counts = (
            early_stage_df['MentalHealthInterview']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        interview_counts.columns = [
            'InterviewResponse',
            'Percentage'
        ]

        interview_counts['InterviewResponse'] = (
            interview_counts['InterviewResponse']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
            })
        )

        fig = px.pie(

            interview_counts,

            names='InterviewResponse',

            values='Percentage',

            color_discrete_sequence=[
                '#ef4444',
                '#facc15',
                '#22c55e'
            ]
        )

        fig.update_traces(



            textfont_size=13,

            hole=0.55,

            pull=[0.025, 0.025, 0.025],

            marker=dict(

                line=dict(
                    color='#0e1117',
                    width=4
                )
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=35,
                r=0,
                t=20,
                b=50
            ),

            showlegend=False,

            annotations=[

                dict(

                    text='Mental<br>Health',

                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='mental_health_interview_chart'
        )

with gr5:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Early Treatment Awareness
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Treatment engagement among individuals identified in early mental health stages"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        treatment_counts = (
            early_stage_df['Treatment']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        treatment_counts.columns = [
            'Treatment',
            'Percentage'
        ]

        treatment_counts['Treatment'] = (
            treatment_counts['Treatment']
            .map({
                0: 'No',
                1: 'Yes'
            })
        )

        fig = px.bar(

            treatment_counts,

            x='Treatment',

            y='Percentage',

            color='Percentage',

            text='Percentage',

            color_continuous_scale='Greens'
        )

        fig.update_traces(

            texttemplate='%{text:.2f}%',

            textposition='outside'
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            xaxis=dict(
                title='Treatment Status',
                showgrid=False
            ),

            yaxis=dict(
                title='Percentage of Individuals',

                range=[
                    0,
                    treatment_counts['Percentage'].max() * 1.15
                ],

                gridcolor='#2d333b'
            ),

            coloraxis_showscale=False
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='early_treatment_awareness'
        )

st.markdown(
    """
    <h3 style="
        color:white;
        font-size:20px;
        margin-bottom:10px;
    ">
    Secondary Psychological Indicators:
    </h3>
    """,
    unsafe_allow_html=True
)

gr6,gr7,gr8,gr9,gr10=st.columns(5)

with gr6:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Indoor Duration Patterns in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Average early-stage occurrence across different indoor duration categories
                0-> 0 days
                1-> 1-14 days
                2-> 15-30 days
                3-> 31-60 days
                4-> 60+ days"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        indoor_counts = (
            early_stage_df['DaysIndoors']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        indoor_counts.columns = [
            'DaysIndoors',
            'Percentage'
        ]
        indoor_counts = indoor_counts.sort_values(
                        by='DaysIndoors'
                        )

        fig = px.line(

            indoor_counts,

            x='DaysIndoors',

            y='Percentage',

            markers=True
        )

        fig.update_traces(

            line=dict(
                width=2.5,
                color="#ff8528"
            ),

            marker=dict(
                size=7,
                color='#93c5fd'
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=0,
                r=0,
                t=35,
                b=30
            ),

            xaxis=dict(
                title='Days Indoors',
                showgrid=False
            ),

            yaxis=dict(
                title='Percentage of Individuals',

                gridcolor='#2d333b'
            )
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='early_stage_daysindoors'
        )
        
with gr7:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Habit Changes in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Habit change patterns among early-stage individuals"
                style="
                color:#9ca3af;
                font-size:15px;
                text-align:right;
                cursor:pointer;
                margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        habits_counts = (
            early_stage_df['HabitsChange']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        habits_counts.columns = [
            'Changes_Habits',
            'Percentage'
        ]

        habits_counts['Changes_Habits'] = (
            habits_counts['Changes_Habits']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
            })
        )

        fig = px.pie(

            habits_counts,

            names='Changes_Habits',

            values='Percentage',

            color_discrete_sequence=[
                "#df2424",
                "#fad12d",
                "#74e91b"
            ]
        )

        fig.update_traces(



            textfont_size=14,

            hole=0.50,

            pull=[0.025, 0.025, 0.025],

            marker=dict(

                line=dict(
                    color='#0e1117',
                    width=4
                )
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=50
            ),

            showlegend=False,

            annotations=[

                dict(

                    text='Habits<br>change',

                    x=0.5,

                    y=0.5,

                    font_size=15,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='habit_changes_early_stage'
        )
        
with gr8:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Mood Swings in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Mood swing patterns among early-stage individuals"
                style="
                    color:#9ca3af;
                    font-size:15px;
                    text-align:right;
                    cursor:pointer;
                    margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        mood_counts = (
            early_stage_df['MoodSwings']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        mood_counts.columns = [
            'MoodSwings',
            'Percentage'
        ]

        mood_counts['MoodSwings'] = (
            mood_counts['MoodSwings']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
            })
        )

        fig = px.pie(

            mood_counts,

            names='MoodSwings',

            values='Percentage',

            color_discrete_sequence=[
                "#df2424",
                '#74e91b',
                '#fad12d'
            ]
        )

        fig.update_traces(



            textfont_size=14,

            hole=0.5,

            pull=[0.025, 0.025, 0.025],

            marker=dict(

                line=dict(
                    color='#0e1117',
                    width=4
                )
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=50
            ),

            showlegend=False,

            annotations=[

                dict(

                    text='Mood',

                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='moodswings_early_stage'
        )

with gr9:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Social Weakness in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Social weakness patterns among early-stage individuals"
                style="
                    color:#9ca3af;
                    font-size:15px;
                    text-align:right;
                    cursor:pointer;
                    margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        social_counts = (
            early_stage_df['SocialWeakness']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        social_counts.columns = [
            'Social_Weakness',
            'Percentage'
        ]

        social_counts['Social_Weakness'] = (
            social_counts['Social_Weakness']
            .map({
                0: 'No',
                1: 'Maybe',
                2: 'Yes'
            })
        )

        fig = px.pie(

            social_counts,

            names='Social_Weakness',

            values='Percentage',

            color_discrete_sequence=[
                "#f7fb2f",
                "#50e736",
                "#d82d2d"
            ]
        )

        fig.update_traces(



            textfont_size=14,

            hole=0.50,

            pull=[0.025, 0.025, 0.025],

            marker=dict(

                line=dict(
                    color='#0e1117',
                    width=4
                )
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=50
            ),

            showlegend=False,

            annotations=[

                dict(

                    text='Social',

                    x=0.5,

                    y=0.5,

                    font_size=18,

                    font_color='white',

                    showarrow=False
                )
            ]
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='social_weakness_early_stage'
        )
        
with gr10:

    with st.container():

        title_col, info_col = st.columns([9,1])

        with title_col:

            st.markdown(
                """
                <h4 style="
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    margin-bottom:0px;
                ">
                Work Interest Patterns in Early Stage Cases
                </h4>
                """,
                unsafe_allow_html=True
            )

        with info_col:

            st.markdown(
                """
                <p title="Work interest distribution among early-stage individuals"
                style="
                    color:#9ca3af;
                    font-size:15px;
                    text-align:right;
                    cursor:pointer;
                    margin-top:2px;
                ">
                ⓘ
                </p>
                """,
                unsafe_allow_html=True
            )

        early_stage_df = filtered_df[
            filtered_df['Early stage'] == 1
        ]

        work_counts = (
            early_stage_df['WorkInterest']
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        work_counts.columns = [
            'WorkInterest',
            'Percentage'
        ]

        work_counts['WorkInterest'] = (
            work_counts['WorkInterest']
            .astype(int)
        )

        work_counts = work_counts.sort_values(
            by='WorkInterest'
        )

        fig = px.line(

            work_counts,

            x='WorkInterest',

            y='Percentage',

            markers=True
        )

        fig.update_traces(

            line=dict(
                width=3,
                color='#f97316'
            ),

            marker=dict(
                size=9,
                color="#74a2fd"
            )
        )

        fig.update_layout(

            height=260,

            paper_bgcolor='#0e1117',

            plot_bgcolor='#0e1117',

            font=dict(
                color='white',
                size=13
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),

            xaxis=dict(
                title='Work Interest Index',
                showgrid=False,
                tickmode='linear',
                dtick=1
            ),

            yaxis=dict(
                title='Percentage of Individuals',

                gridcolor='#2d333b'
            )
        )

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                'displayModeBar': False
            },

            key='workinterest_early_stage'
        )
        





@st.dialog("📊 Assessment Report")
def prediction_report():

    st.markdown(
        "## 📊 Personal Assessment Report"
    )




    st.markdown(
            "## 🔥 Stress Analysis"
        )

    stress_avg = df["StressScore"].mean()

    fig = go.Figure()

    fig.add_bar(

            y=["Population Average"],

            x=[stress_avg],

            orientation="h",

            name="Population"

        )

    fig.add_bar(

            y=["Your Score"],

            x=[st.session_state.stress_score],

            orientation="h",

            name="You"

        )

    fig.update_layout(

            height=250,

            paper_bgcolor="#0b1220",

            plot_bgcolor="#0b1220",

            font_color="white",

            barmode="group",

            xaxis_title="Stress Score"

        )

    st.plotly_chart(
            fig,
            use_container_width=True,
            key="stress_compare"
        )

    st.metric(
            "Your Stress Score",
            round(
                st.session_state.stress_score,
                2
            )
        )

    st.markdown("---")

    st.markdown(
    "## 🏠 Isolation Analysis"
        )

    isolation_avg = df[
        "Isolation_env"].mean()

    fig = go.Figure()

    fig.add_bar(

    y=["Population Average"],

    x=[isolation_avg],

    orientation="h",

    name="Population"

    )

    fig.add_bar(

        y=["Your Score"],

        x=[st.session_state.isolation_env],

        orientation="h",

        name="You"

    )

    fig.update_layout(

        height=250,

        paper_bgcolor="#0b1220",

        plot_bgcolor="#0b1220",

        font_color="white",

        barmode="group",

        xaxis_title="Isolation Score"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="isolation_compare"
    )

    st.metric(
        "Your Isolation Score",
        round(
            st.session_state.isolation_env,
            2
        )
    )



    

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=st.session_state.confidence,
            title={
                "text":
                "Likelihood of Seeking Support"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                },
                "bar":{
                    "color":"#ef4444"
                }
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="#0b1220",
        font_color="white",
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### Where You Stand"
    )

    fig = px.box(
        df,
        y="StressScore",
        points=False
    )

    fig.add_scatter(
        y=[st.session_state.stress_score],
        x=[0],
        mode="markers",
        marker=dict(
            size=16,
            color="red"
        ),
        name="You"
    )

    fig.update_layout(
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font_color="white",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    fig = px.box(
        df,
        y="Isolation_env",
        points=False
    )

    fig.add_scatter(
        y=[st.session_state.isolation_env],
        x=[0],
        mode="markers",
        marker=dict(
            size=16,
            color="red"
        ),
        name="You"
    )

    fig.update_layout(
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font_color="white",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )




    
if st.session_state.get(
    "show_report",
    False
):

    st.session_state.show_report = False

    prediction_report()

if "open_chat" not in st.session_state:

    st.session_state.open_chat = False   

@st.dialog(
    "🤖I'm here to hear You ",
    width="large"
    )
def chatbot_popup():

        if "messages" not in st.session_state:

            st.session_state.messages = []

        assessment_exists = bool(
            st.session_state.get(
                "assessment_context",
                {}
            )
        )

        if not assessment_exists:

            st.warning(
                """
                You have not completed the Mental Health
                Assessment yet.

                Complete the assessment for personalized
                guidance and insights.
                """
            )

        else:

            st.success(
                "Personalized guidance enabled ✓"
            )

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        user_message = st.chat_input(
            "Tell me what's on your mind..."
        )

        if user_message:

            st.session_state.messages.append(
                {
                    "role":"user",
                    "content":user_message
                }
            )

            assessment_context = (
                st.session_state.get(
                    "assessment_context",
                    {}
                )
            )

            response = generate_response(

                user_message,

                assessment_context,

                st.session_state.messages

            )

            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":response
                }
            )

            st.rerun()
if st.session_state.open_chat:

    chatbot_popup()
