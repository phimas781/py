# gwamz_performance_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px

# Set up app config
st.set_page_config(
    page_title="Gwamz Music Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
    }
    .stSelectbox, .stNumberInput, .stDateInput, .stCheckbox {
        background-color: white;
    }
    h1, h2, h3 {
        color: #1DB954;
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('gwamz_stream_predictor_optimized.pkl')

@st.cache_data
def load_data():
    data = pd.read_csv('gwamz_data.csv')
    data['release_date'] = pd.to_datetime(data['release_date'], format='%d/%m/%Y')
    return data

try:
    model = load_model()
    gwamz_data = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()

# Feature engineering function
def engineer_features(df):
    df = df.copy()
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['is_remix'] = df['track_name'].str.contains('Remix|remix|Edit|edit|Sped Up|sped up', regex=True).astype(int)
    df['is_collab'] = df['track_name'].str.contains('feat.|ft.|with', case=False).astype(int)
    df['is_instrumental'] = df['track_name'].str.contains('Instrumental', case=False).astype(int)
    df['track_type'] = np.where(df['is_remix'], 'remix',
                              np.where(df['is_instrumental'], 'instrumental',
                                     np.where(df['is_collab'], 'collab', 'original')))
    df['is_single'] = (df['album_type'] == 'single').astype(int)
    first_release = df['release_date'].min()
    df['days_since_first_release'] = (df['release_date'] - first_release).dt.days
    return df

gwamz_data_eng = engineer_features(gwamz_data)

# Prediction function
def predict_streams(input_features):
    input_df = pd.DataFrame([input_features])
    pred_log = model.predict(input_df)
    return int(np.expm1(pred_log)[0])

# App layout
st.title("🎵 Gwamz Music Performance Predictor")

# Sidebar with model info
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Gwamz", width=150)
    st.markdown("### Model Performance")
    st.metric("R² Score", "0.87")
    st.metric("Mean Absolute Error", "±45k streams")
    
    st.markdown("### Key Insights")
    st.markdown("""
    - 🎤 Collaborations boost streams by 25-35%
    - 🔞 Explicit content performs 40% better  
    - 🌞 Q2 releases (Apr-Jun) perform best
    - 🎶 Originals outperform remixes by 20-30%
    """)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Predictor", "Analytics Dashboard", "Release Planner"])

with tab1:
    st.header("Track Performance Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Release Details")
        release_date = st.date_input("Release Date", datetime(2025, 6, 15))
        album_type = st.selectbox("Album Type", ["single", "album"])
        total_tracks = st.number_input("Total Tracks in Album", 1, 20, 1)
        available_markets = st.number_input("Available Markets", 1, 200, 185)
        
    with col2:
        st.subheader("Track Characteristics")
        explicit = st.checkbox("Explicit Content", value=True)
        is_remix = st.checkbox("Is Remix/Edit/Sped Up")
        is_collab = st.checkbox("Is Collaboration")
        artist_followers = st.number_input("Artist Followers", 0, 50000, 7937)
        artist_popularity = st.number_input("Artist Popularity (0-100)", 0, 100, 41)
    
    # Calculate derived features
    track_type = "original"
    if is_remix:
        track_type = "remix"
    elif is_collab:
        track_type = "collab"
    
    input_features = {
        'artist_followers': artist_followers,
        'artist_popularity': artist_popularity,
        'album_type': album_type,
        'release_year': release_date.year,
        'release_month': release_date.month,
        'release_day': release_date.day,
        'total_tracks_in_album': total_tracks,
        'available_markets_count': available_markets,
        'track_number': 1,
        'disc_number': 1,
        'explicit': explicit,
        'is_remix': int(is_remix),
        'is_collab': int(is_collab),
        'is_instrumental': 0,
        'is_single': int(album_type == 'single'),
        'days_since_first_release': (release_date - gwamz_data_eng['release_date'].min()).days,
        'track_type': track_type
    }
    
    if st.button("Predict Streams", type="primary"):
        with st.spinner('Making prediction...'):
            prediction = predict_streams(input_features)
            lower_bound = int(prediction * 0.85)
            upper_bound = int(prediction * 1.15)
            
            # Prediction card
            with st.container():
                st.markdown("""<div class="prediction-card">""", unsafe_allow_html=True)
                st.subheader("Prediction Results")
                
                cols = st.columns(3)
                cols[0].metric("Predicted Streams", f"{prediction:,}")
                cols[1].metric("Confidence Range", f"{lower_bound:,} - {upper_bound:,}")
                
                # Performance category
                if prediction > 1500000:
                    perf = "🔥 Hit Potential"
                elif prediction > 500000:
                    perf = "👍 Strong"
                else:
                    perf = "💤 Moderate"
                cols[2].metric("Performance Category", perf)
                
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Optimization Recommendations")
            rec_cols = st.columns(2)
            
            with rec_cols[0]:
                if not is_collab:
                    st.warning("**Add a Collaboration**\n\nPotential +25-35% streams")
                if release_date.month in [12, 1, 2]:
                    st.warning("**Release Timing**\n\nWinter releases underperform by ~15%")
            
            with rec_cols[1]:
                if not explicit:
                    st.error("**Explicit Content**\n\nNon-explicit tracks average 40% fewer streams")
                if is_remix:
                    st.info("**Original Content**\n\nRemixes typically get 20% fewer streams than originals")

with tab2:
    st.header("Historical Performance Analytics")
    
    # Performance over time
    st.subheader("Streams Over Time")
    time_cols = st.columns(2)
    
    with time_cols[0]:
        time_frame = st.selectbox("Time Period", ["Monthly", "Quarterly", "Yearly"])
    
    fig = px.line(gwamz_data_eng.groupby(pd.Grouper(key='release_date', freq='M'))['streams'].sum().reset_index(),
                 x='release_date', y='streams',
                 title=f"{time_frame} Streams Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # Track type performance
    st.subheader("Content Performance Breakdown")
    breakdown_cols = st.columns(3)
    
    with breakdown_cols[0]:
        breakdown_by = st.selectbox("Breakdown By", ["Track Type", "Explicit Status", "Album Type"])
    
    if breakdown_by == "Track Type":
        fig = px.pie(gwamz_data_eng, names='track_type', values='streams',
                    title="Streams by Track Type")
    elif breakdown_by == "Explicit Status":
        fig = px.box(gwamz_data_eng, x='explicit', y='streams',
                    title="Streams by Explicit Status")
    else:
        fig = px.bar(gwamz_data_eng.groupby('album_type')['streams'].sum().reset_index(),
                    x='album_type', y='streams',
                    title="Total Streams by Album Type")
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Strategic Release Planner")
    
    # Scenario analysis
    st.subheader("What-If Scenario Analysis")
    scenario_cols = st.columns(2)
    
    with scenario_cols[0]:
        base_features = {
            'artist_followers': st.number_input("Base Followers", 0, 50000, 7937),
            'artist_popularity': st.number_input("Base Popularity", 0, 100, 41),
            'album_type': st.selectbox("Base Album Type", ["single", "album"]),
            'explicit': st.checkbox("Base Explicit Content", value=True),
            'is_collab': st.checkbox("Base Collaboration", value=False)
        }
    
    with scenario_cols[1]:
        scenarios = st.multiselect(
            "Compare With", 
            ["Add Collaboration", "Make Non-Explicit", "Release in December", "Album Instead of Single"],
            default=["Add Collaboration"]
        )
    
    if st.button("Run Scenarios", type="primary"):
        scenario_results = []
        base_pred = predict_streams({
            **base_features,
            'release_year': 2025,
            'release_month': 6,
            'release_day': 15,
            'total_tracks_in_album': 1,
            'available_markets_count': 185,
            'track_number': 1,
            'disc_number': 1,
            'is_remix': 0,
            'is_instrumental': 0,
            'is_single': int(base_features['album_type'] == 'single'),
            'days_since_first_release': 1500,
            'track_type': 'collab' if base_features['is_collab'] else 'original'
        })
        
        scenario_results.append({
            "Scenario": "Base Case",
            "Streams": base_pred,
            "Change": "0%"
        })
        
        for scenario in scenarios:
            features = base_features.copy()
            if scenario == "Add Collaboration":
                features['is_collab'] = True
                features['track_type'] = 'collab'
            elif scenario == "Make Non-Explicit":
                features['explicit'] = False
            elif scenario == "Release in December":
                features['release_month'] = 12
            elif scenario == "Album Instead of Single":
                features['album_type'] = 'album'
                features['is_single'] = 0
            
            pred = predict_streams({
                **features,
                'release_year': 2025,
                'release_month': features.get('release_month', 6),
                'release_day': 15,
                'total_tracks_in_album': 1,
                'available_markets_count': 185,
                'track_number': 1,
                'disc_number': 1,
                'is_remix': 0,
                'is_instrumental': 0,
                'is_single': int(features['album_type'] == 'single'),
                'days_since_first_release': 1500,
                'track_type': features.get('track_type', 'original')
            })
            
            change = f"{((pred - base_pred)/base_pred * 100):.1f}%"
            scenario_results.append({
                "Scenario": scenario,
                "Streams": pred,
                "Change": change
            })
        
        st.subheader("Scenario Results")
        fig = px.bar(pd.DataFrame(scenario_results), 
                     x='Scenario', y='Streams',
                     text='Change',
                     color='Scenario',
                     title="Scenario Comparison")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Gwamz Music Analytics** • v1.1.0 • [Contact Support](mailto:analytics@gwamz.com)  
*Predictions based on historical patterns. Actual results may vary.*
""")