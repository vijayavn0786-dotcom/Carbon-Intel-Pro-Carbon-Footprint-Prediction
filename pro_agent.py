import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="CarbonIntel Pro V2.7", page_icon="🌍", layout="wide")

# --- UI STYLING (THEME & VISIBILITY) ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    
    /* SIDEBAR styling */
    [data-testid="stSidebar"] { background-color: #1e3932 !important; }
    [data-testid="stSidebar"] * { color: white !important; }

    /* METRIC BOXES: Black text for visibility */
    [data-testid="stMetricLabel"] > div { color: #000000 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] > div { color: #000000 !important; font-weight: 800 !important; }
    
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-left: 5px solid #2e7d32;
    }

    .stButton>button {
        background-color: #2e7d32;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    
    h1, h2, h3 { color: #1e3932; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & AI CORE ---
@st.cache_data
def load_and_clean():
    if not os.path.exists('Carbon Emission.csv'): return None
    df = pd.read_csv('Carbon Emission.csv')
    df['Vehicle Type'] = df['Vehicle Type'].fillna('None/N-A')
    return df

@st.cache_resource
def train_agent_core(df):
    ml_df = df.copy()
    encoders = {}
    cat_cols = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 
                'Transport', 'Vehicle Type', 'Social Activity', 'Frequency of Traveling by Air', 
                'Waste Bag Size', 'Energy efficiency', 'Recycling', 'Cooking_With']
    
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le

    X = ml_df.drop(columns=['CarbonEmission'])
    y = ml_df['CarbonEmission']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    feat_imp_series = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    
    return model, encoders, X.columns, feat_imp_series

# --- APPLICATION ---
df = load_and_clean()

if df is not None:
    avg_e = df['CarbonEmission'].mean() 
    model, encoders, feature_names, global_feat_imp = train_agent_core(df)
    
    st.sidebar.title("🌍 EcoAgent Pro")
    st.sidebar.markdown("---")
    nav = st.sidebar.radio("Navigation Menu", ["Executive Dashboard", "Predictive Agent", "Data Explorer"])
    
    if nav == "Executive Dashboard":
        st.title("Sustainability Intelligence Dashboard")
        
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Avg Monthly Emission", value=f"{avg_e:.0f} kg", delta="Global Benchmark")
        m2.metric(label="Total Records", value=f"{len(df):,}", delta="Verified")
        m3.metric(label="Analysis Mode", value="Agentic AI", delta="Active")
        
        st.markdown("---")
        
        c1, col_imp = st.columns([1, 1])
        with c1:
            st.subheader("Emission by Diet Type")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Diet', y='CarbonEmission', data=df, hue='Diet', palette='Greens_d', ax=ax1, legend=False)
            st.pyplot(fig1)
        
        with col_imp:
            st.subheader("Key Emission Drivers (Top 8)")
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            global_feat_imp.tail(8).plot(kind='barh', color='#2e7d32', ax=ax_imp)
            st.pyplot(fig_imp)

    elif nav == "Predictive Agent":
        st.title("🤖 Agentic Prediction Engine")
        st.write("Simulate lifestyle changes to see real-time carbon impacts.")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                b_type = st.selectbox("Body Type", df['Body Type'].unique())
                sex = st.selectbox("Sex", df['Sex'].unique())
                diet = st.selectbox("Diet", df['Diet'].unique())
                travel = st.selectbox("Air Travel Frequency", df['Frequency of Traveling by Air'].unique())
            with col2:
                trans = st.selectbox("Primary Transport", df['Transport'].unique())
                v_type = st.selectbox("Vehicle Type", df['Vehicle Type'].unique())
                dist = st.number_input("Monthly KM", 0, 10000, 500)
                heat = st.selectbox("Heating Source", df['Heating Energy Source'].unique())
            with col3:
                waste_size = st.selectbox("Waste Bag Size", df['Waste Bag Size'].unique())
                waste_count = st.slider("Weekly Bags", 1, 10, 3)
                grocery = st.number_input("Monthly Grocery ($)", 50, 500, 200)
                efficiency = st.selectbox("Home Efficiency", df['Energy efficiency'].unique())

            if st.button("RUN AGENTIC AUDIT"):
                sim_data = {
                    'Body Type': b_type, 'Sex': sex, 'Diet': diet, 'How Often Shower': 'daily',
                    'Heating Energy Source': heat, 'Transport': trans, 'Vehicle Type': v_type,
                    'Social Activity': 'often', 'Monthly Grocery Bill': grocery,
                    'Frequency of Traveling by Air': travel, 'Vehicle Monthly Distance Km': dist,
                    'Waste Bag Size': waste_size, 'Waste Bag Weekly Count': waste_count,
                    'How Long TV PC Daily Hour': 5, 'How Many New Clothes Monthly': 2,
                    'How Long Internet Daily Hour': 5, 'Energy efficiency': efficiency,
                    'Recycling': "['Metal']", 'Cooking_With': "['Stove']"
                }

                encoded_features = []
                for col in feature_names:
                    if col in encoders:
                        encoded_features.append(encoders[col].transform([str(sim_data[col])])[0])
                    else:
                        encoded_features.append(sim_data[col])
                
                prediction = model.predict([encoded_features])[0]
                
                st.markdown("---")
                # REPORT SECTION 1: Predicted Output
                st.subheader("📊 Predicted Monthly Output")
                st.metric(label="Carbon Emission Forecast", value=f"{prediction:.2f} kg", delta=f"{prediction - avg_e:.2f} vs Avg", delta_color="inverse")
                
                col_rep1, col_rep2 = st.columns(2)
                
                with col_rep1:
                    # REPORT SECTION 2: Agent Intelligence
                    st.subheader("📋 Agent Intelligence Report")
                    if prediction > avg_e:
                        st.error(f"High Impact Profile: Carbon output is {(prediction/avg_e - 1)*100:.1f}% higher than average.")
                    else:
                        st.success("Sustainable Profile: Carbon footprint optimized.")
                    
                    top_driver = global_feat_imp.index[-1]
                    st.info(f"**Primary Driver:** Statistical analysis identifies '{top_driver}' as the strongest influencer of your current score.")

                with col_rep2:
                    # REPORT SECTION 3: Mitigation Suggestions
                    st.subheader("🌿 Mitigation Suggestions")
                    
                    # Logic-based agentic suggestions
                    if travel in ['frequently', 'very frequently']:
                        st.warning("- **Travel:** Air travel is significantly raising your footprint. Consider reducing non-essential flights.")
                    
                    if trans == 'private' and dist > 1000:
                        st.warning("- **Transport:** High private mileage detected. Switching to public transit or carpooling can lower impact.")
                    
                    if diet == 'omnivore':
                        st.info("- **Diet:** Transitioning to a 'Pescatarian' or 'Vegetarian' diet can lower monthly emissions by approx. 15-20%.")
                        
                    if heat in ['coal', 'wood']:
                        st.info("- **Energy:** Solid fuel heating is carbon-intensive. Upgrading to Natural Gas or Solar can save up to 1,000kg/year.")
                    
                    if not any([travel in ['frequently', 'very frequently'], trans == 'private' and dist > 1000, diet == 'omnivore', heat in ['coal', 'wood']]):
                        st.success("- **Optimization:** You already follow many green practices. Focus on waste reduction and energy-efficient appliances.")

    elif nav == "Data Explorer":
        st.title("📊 Data Intelligence Explorer")
        st.markdown("### Export Capability")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 DOWNLOAD FULL DATA REPORT (CSV)", data=csv, file_name='carbon_emission_report.csv', mime='text/csv')
        st.markdown("---")
        st.dataframe(df.head(100), width='stretch')

else:
    st.error("Missing Data: Ensure 'Carbon Emission.csv' is in the project folder.")