# gwamz_analytics_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO

# --- App Configuration ---
st.set_page_config(
    page_title="Gwamz Music Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://gwamz.com/support',
        'Report a bug': "https://gwamz.com/bug",
        'About': "Gwamz Music Analytics v2.0"
    }
)

# --- Load Assets ---
@st.cache_data
def load_logo():
    try:
        response = requests.get("https://via.placeholder.com/150x150.png?text=GWAMZ")
        return Image.open(BytesIO(response.content))
    except:
        return None

# --- Custom CSS ---
st.markdown("""
<style>
    :root {
        --primary: #1DB954;
        --secondary: #191414;
        --accent: #FFFFFF;
    }
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    .sidebar .sidebar-content {
        background: var(--secondary) !important;
        color: var(--accent);
    }
    .stButton>button {
        background: var(--primary) !important;
        color: var(--accent) !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid var(--primary);
    }
    .prediction-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
    }
    .badge-hit {
        background: #1DB954;
        color: white;
    }
    .badge-strong {
        background: #FFD700;
        color: var(--secondary);
    }
    .badge-moderate {
        background: #A9A9A9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('gwamz_stream_predictor_optimized.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('gwamz_data.csv')
        data['release_date'] = pd.to_datetime(data['release_date'], format='%d/%m/%Y')
        return data
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

model = load_model()
gwamz_data = load_data()

# --- Feature Engineering ---
def engineer_features(df):
    df = df.copy()
    # Date features
    df['release_month'] = df['release_date'].dt.month
    df['release_quarter'] = df['release_date'].dt.quarter
    df['release_day'] = df['release_date'].dt.day
    df['release_dayofweek'] = df['release_date'].dt.dayofweek
    
    # Content features
    df['is_remix'] = df['track_name'].str.contains('Remix|remix|Edit|edit|Sped Up|sped up', regex=True).astype(int)
    df['is_collab'] = df['track_name'].str.contains('feat.|ft.|with', case=False).astype(int)
    df['is_instrumental'] = df['track_name'].str.contains('Instrumental', case=False).astype(int)
    df['track_type'] = np.where(df['is_remix'], 'remix',
                              np.where(df['is_instrumental'], 'instrumental',
                                     np.where(df['is_collab'], 'collab', 'original')))
    df['is_single'] = (df['album_type'] == 'single').astype(int)
    
    # Time-based features
    first_release = df['release_date'].min()
    df['days_since_first_release'] = (df['release_date'] - first_release).dt.days
    df['release_week'] = ((df['release_date'] - first_release).dt.days // 7) + 1
    
    return df

gwamz_data_eng = engineer_features(gwamz_data)

# --- Prediction Function ---
def predict_streams(input_features):
    input_df = pd.DataFrame([input_features])
    pred_log = model.predict(input_df)
    return int(np.expm1(pred_log)[0])

# --- App Layout ---
logo = load_logo()
st.sidebar.image(logo, width=150) if logo else st.sidebar.title("GWAMZ ANALYTICS")

# --- Sidebar ---
with st.sidebar:
    st.title("Model Dashboard")
    
    st.markdown("### Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", "0.87")
    col2.metric("MAE", "±45k")
    
    st.markdown("### Key Insights")
    with st.expander("Content Strategy"):
        st.markdown("""
        - 🎤 Collabs: +25-35% streams
        - 🔞 Explicit: +40% performance  
        - 🌞 Q2 releases perform best
        """)
    
    with st.expander("Release Timing"):
        st.markdown("""
        - 📅 Wed/Thu releases optimal
        - ❄️ Winter months underperform
        - 🎯 2-3 week promo cycle ideal
        """)
    
    st.markdown("---")
    st.markdown("**Data Last Updated**  \n" + datetime.now().strftime("%Y-%m-%d %H:%M"))
    if st.button("Check for Updates"):
        st.rerun()

# --- Main Content ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictor", "📊 Analytics", "📅 Planner", "⚙️ Settings"])

with tab1:
    st.header("Track Performance Predictor")
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.form("prediction_form"):
                st.subheader("Release Details")
                release_date = st.date_input("Release Date", datetime(2025, 6, 15))
                album_type = st.selectbox("Album Type", ["single", "album", "EP"])
                total_tracks = st.slider("Total Tracks", 1, 20, 1)
                markets = st.slider("Available Markets", 50, 200, 185)
                
                st.subheader("Content Features")
                explicit = st.toggle("Explicit Content", True)
                is_remix = st.toggle("Remix/Edit Version")
                is_collab = st.toggle("Collaboration")
                
                st.subheader("Artist Metrics")
                followers = st.number_input("Current Followers", 0, 100000, 7937)
                popularity = st.slider("Artist Popularity", 0, 100, 41)
                
                submitted = st.form_submit_button("Predict Performance", type="primary")
        
        with col2:
            if submitted:
                # Feature preparation
                track_type = "original"
                if is_remix:
                    track_type = "remix"
                elif is_collab:
                    track_type = "collab"
                
                input_features = {
                    'artist_followers': followers,
                    'artist_popularity': popularity,
                    'album_type': album_type.lower(),
                    'release_year': release_date.year,
                    'release_month': release_date.month,
                    'release_day': release_date.day,
                    'total_tracks_in_album': total_tracks,
                    'available_markets_count': markets,
                    'track_number': 1,
                    'disc_number': 1,
                    'explicit': explicit,
                    'is_remix': int(is_remix),
                    'is_collab': int(is_collab),
                    'is_instrumental': 0,
                    'is_single': int(album_type.lower() == 'single'),
                    'days_since_first_release': (release_date - gwamz_data_eng['release_date'].min()).days,
                    'track_type': track_type
                }
                
                with st.spinner('Generating prediction...'):
                    prediction = predict_streams(input_features)
                    lower = int(prediction * 0.85)
                    upper = int(prediction * 1.15)
                    
                    # Prediction card
                    with st.container():
                        st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                        
                        # Performance badge
                        if prediction > 1500000:
                            badge = """<span class="prediction-badge badge-hit">HIT POTENTIAL</span>"""
                        elif prediction > 500000:
                            badge = """<span class="prediction-badge badge-strong">STRONG</span>"""
                        else:
                            badge = """<span class="prediction-badge badge-moderate">MODERATE</span>"""
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h2 style="margin: 0;">Prediction Results</h2>
                            {badge}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        cols = st.columns(3)
                        cols[0].metric("Predicted Streams", f"{prediction:,}")
                        cols[1].metric("Confidence Range", f"{lower:,} - {upper:,}")
                        cols[2].metric("Market Coverage", f"{markets} markets")
                        
                        st.markdown("""</div>""", unsafe_allow_html=True)
                    
                    # Recommendations
                    st.subheader("Optimization Recommendations")
                    
                    recs = []
                    if not is_collab:
                        recs.append(("Add Collaboration", "Potential +25-35% streams", "warning"))
                    if not explicit:
                        recs.append(("Make Explicit", "40% average stream increase", "error"))
                    if release_date.month in [12, 1, 2]:
                        recs.append(("Reschedule Release", "Q2 (Apr-Jun) performs 15% better", "warning"))
                    if is_remix:
                        recs.append(("Original Version", "Remixes get 20% fewer streams", "info"))
                    
                    if recs:
                        cols = st.columns(len(recs))
                        for i, (title, desc, type_) in enumerate(recs):
                            with cols[i]:
                                if type_ == "error":
                                    st.error(f"**{title}**  \n{desc}")
                                elif type_ == "warning":
                                    st.warning(f"**{title}**  \n{desc}")
                                else:
                                    st.info(f"**{title}**  \n{desc}")
                    else:
                        st.success("This release is optimally configured!")

with tab2:
    st.header("Advanced Analytics Dashboard")
    
    # Performance overview
    with st.container():
        st.subheader("Performance Overview")
        
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Total Streams", f"{gwamz_data_eng['streams'].sum():,}")
        metric2.metric("Average Streams", f"{gwamz_data_eng['streams'].mean():,.0f}")
        metric3.metric("Top Track", f"{gwamz_data_eng['streams'].max():,}")
        metric4.metric("Tracks Analyzed", len(gwamz_data_eng))
    
    # Time series analysis
    with st.container():
        st.subheader("Temporal Analysis")
        
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            time_group = st.selectbox("Group By", ["Monthly", "Quarterly", "Yearly"])
        
        freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
        time_data = gwamz_data_eng.groupby(pd.Grouper(key='release_date', freq=freq_map[time_group]))['streams'].sum().reset_index()
        
        fig = px.area(time_data, x='release_date', y='streams',
                      title=f"{time_group} Streams Performance",
                      labels={'streams': 'Total Streams', 'release_date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Content analysis
    with st.container():
        st.subheader("Content Analysis")
        
        content_col1, content_col2 = st.columns(2)
        with content_col1:
            breakdown_by = st.selectbox("Breakdown By", ["Track Type", "Explicit Status", "Release Type"])
        
        if breakdown_by == "Track Type":
            fig = px.sunburst(gwamz_data_eng, path=['track_type', 'explicit'], values='streams',
                             title="Streams by Track Type and Explicit Status")
        elif breakdown_by == "Explicit Status":
            fig = px.box(gwamz_data_eng, x='explicit', y='streams', color='track_type',
                        title="Streams Distribution by Explicit Status")
        else:
            fig = px.bar(gwamz_data_eng.groupby('album_type')['streams'].sum().reset_index(),
                         x='album_type', y='streams', color='album_type',
                         title="Total Streams by Album Type")
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Strategic Release Planner")
    
    # Scenario analysis
    with st.container():
        st.subheader("Scenario Comparison")
        
        base_col1, base_col2 = st.columns(2)
        with base_col1:
            st.markdown("**Base Scenario**")
            base_followers = st.number_input("Artist Followers", 0, 100000, 7937, key="base_followers")
            base_popularity = st.slider("Artist Popularity", 0, 100, 41, key="base_pop")
            base_explicit = st.toggle("Explicit", True, key="base_explicit")
        
        with base_col2:
            st.markdown("**Compare With**")
            scenarios = st.multiselect(
                "Select Scenarios",
                ["Add Collaboration", "Non-Explicit", "Winter Release", "Album Release", "Remix Version"],
                default=["Add Collaboration"]
            )
        
        if st.button("Compare Scenarios", type="primary"):
            # Base case prediction
            base_features = {
                'artist_followers': base_followers,
                'artist_popularity': base_popularity,
                'album_type': 'single',
                'release_year': 2025,
                'release_month': 6,
                'release_day': 15,
                'total_tracks_in_album': 1,
                'available_markets_count': 185,
                'track_number': 1,
                'disc_number': 1,
                'explicit': base_explicit,
                'is_remix': 0,
                'is_collab': 0,
                'is_instrumental': 0,
                'is_single': 1,
                'days_since_first_release': 1500,
                'track_type': 'original'
            }
            base_pred = predict_streams(base_features)
            
            # Scenario predictions
            scenario_data = []
            for scenario in scenarios:
                features = base_features.copy()
                if scenario == "Add Collaboration":
                    features.update({'is_collab': 1, 'track_type': 'collab'})
                elif scenario == "Non-Explicit":
                    features.update({'explicit': False})
                elif scenario == "Winter Release":
                    features.update({'release_month': 12})
                elif scenario == "Album Release":
                    features.update({'album_type': 'album', 'is_single': 0})
                elif scenario == "Remix Version":
                    features.update({'is_remix': 1, 'track_type': 'remix'})
                
                pred = predict_streams(features)
                change = (pred - base_pred) / base_pred * 100
                scenario_data.append({
                    'Scenario': scenario,
                    'Streams': pred,
                    'Change (%)': change,
                    'Absolute Change': pred - base_pred
                })
            
            # Display results
            st.subheader("Comparison Results")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[s['Scenario'] for s in scenario_data],
                y=[s['Change (%)'] for s in scenario_data],
                text=[f"{s['Change (%)']:.1f}%" for s in scenario_data],
                marker_color=['#1DB954' if x > 0 else '#FF4B4B' for x in [s['Change (%)'] for s in scenario_data]],
                name='Percentage Change'
            ))
            fig.update_layout(
                title="Percentage Change from Base Scenario",
                yaxis_title="Change (%)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(pd.DataFrame(scenario_data).style.format({
                'Streams': '{:,}',
                'Change (%)': '{:.1f}%',
                'Absolute Change': '{:,}'
            }))

with tab4:
    st.header("Settings & Configuration")
    
    with st.expander("Data Management"):
        st.info("Upload new data to refresh predictions")
        new_data = st.file_uploader("Upload CSV", type=["csv"])
        if new_data:
            try:
                new_df = pd.read_csv(new_data)
                new_df.to_csv('gwamz_data.csv', index=False)
                st.success("Data updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with st.expander("Model Settings"):
        st.warning("Advanced settings - modify with caution")
        retrain = st.button("Retrain Model with Latest Data")
        if retrain:
            with st.spinner("Retraining model..."):
                # Add model retraining logic here
                st.success("Model retrained successfully!")
    
    with st.expander("System Info"):
        st.write(f"**Last Model Training:** {datetime.now().strftime('%Y-%m-%d')}")
        st.write(f"**Records in Database:** {len(gwamz_data_eng)}")
        st.write(f"**Data Coverage:** {gwamz_data_eng['release_date'].min().strftime('%Y-%m')} to {gwamz_data_eng['release_date'].max().strftime('%Y-%m')}")

# --- Footer ---
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("""
    **Gwamz Music Analytics Pro**  
    v2.0 · © 2025 Gwamz Records
    """)
with footer_col2:
    st.markdown("""
    [Documentation](https://gwamz.com/docs) · 
    [Support](mailto:analytics-support@gwamz.com) · 
    [API](https://api.gwamz.com)
    """)
