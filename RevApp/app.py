import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

st.set_page_config(page_title="GATEWAYS 2025", layout="wide")

df = pd.read_csv("fest_dataset.csv")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#020617);
}
.header {
    font-size:40px;
    font-weight:800;
    background: linear-gradient(90deg,#3b82f6,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub {
    color:#94a3b8;
    margin-bottom:20px;
}
.card {
    background: linear-gradient(135deg,#1e293b,#020617);
    padding:20px;
    border-radius:14px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0px 8px 25px rgba(59,130,246,0.3);
}
h3 { color:#38bdf8; }
h2 { color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>GATEWAYS 2025 Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Participation • Analysis • Feedback</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Dashboard","Participation","Feedback"])

events = st.sidebar.multiselect("Event", df['Event Name'].unique(), df['Event Name'].unique())
states = st.sidebar.multiselect("State", df['State'].unique(), df['State'].unique())

data = df[(df['Event Name'].isin(events)) & (df['State'].isin(states))]

# Dashboard
with tab1:
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='card'><h3>Participants</h3><h2>{data.shape[0]}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h3>Events</h3><h2>{data['Event Name'].nunique()}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h3>Avg Rating</h3><h2>{round(data['Rating'].mean(),2)}</h2></div>", unsafe_allow_html=True)

    st.subheader("State-wise Participation")

    state_counts = data['State'].value_counts()

    fig, ax = plt.subplots()
    state_counts.plot(kind='bar', ax=ax)

    st.pyplot(fig)

# Participation
with tab2:
    col1,col2 = st.columns(2)

    fig1, ax1 = plt.subplots()
    data['Event Name'].value_counts().plot(kind='bar', ax=ax1)
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    data['College'].value_counts().head(10).plot(kind='bar', ax=ax2)
    col2.pyplot(fig2)

# Feedback
with tab3:
    sia = SentimentIntensityAnalyzer()

    data['Sentiment'] = data['Feedback on Fest'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )

    col3,col4 = st.columns(2)

    fig3, ax3 = plt.subplots()
    data['Sentiment'].plot(kind='hist', ax=ax3)
    col3.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    data['Rating'].plot(kind='hist', ax=ax4)
    col4.pyplot(fig4)

    st.subheader("Dataset")
    st.dataframe(data)
