import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Complaint Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    padding-bottom:0rem;
    max-width:95%;
}

.main{
    background:#050816;
}

section[data-testid="stSidebar"]{
    display:none;
}

h1,h2,h3,h4,p{
    color:white;
}

/* KPI CARDS */

[data-testid="stMetric"]{
    background:#111827;
    border:1px solid #1f2937;
    padding:15px;
    border-radius:14px;
    text-align:center;
}

/* INSIGHT BOX */

.insight-box{
    background:#111827;
    padding:16px;
    border-radius:14px;
    border:1px solid #1f2937;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv("DATA/GPT_reviews.csv")

# ------------------------------------------------
# SENTIMENT ANALYSIS
# ------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    score = analyzer.polarity_scores(str(text))

    if score["compound"] >= 0.05:
        return "Positive"

    elif score["compound"] <= -0.05:
        return "Negative"

    else:
        return "Neutral"

df["sentiment"] = df["Comment"].apply(get_sentiment)

# ------------------------------------------------
# METRICS
# ------------------------------------------------

total_reviews = len(df)

positive_reviews = len(
    df[df["sentiment"] == "Positive"]
)

negative_reviews = len(
    df[df["sentiment"] == "Negative"]
)

neutral_reviews = len(
    df[df["sentiment"] == "Neutral"]
)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.markdown("""
<h1 style='font-size:42px;margin-bottom:0px;'>
🧠 AI Complaint Intelligence Platform
</h1>

<p style='font-size:16px;color:#9ca3af;margin-top:0px;'>
AI sentiment analytics + complaint intelligence dashboard
</p>
""", unsafe_allow_html=True)

# ------------------------------------------------
# KPI ROW
# ------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="Total Reviews",
        value=f"{total_reviews:,}"
    )

with c2:
    st.metric(
        label="Positive",
        value=f"{positive_reviews:,}"
    )

with c3:
    st.metric(
        label="Negative",
        value=f"{negative_reviews:,}"
    )

with c4:
    st.metric(
        label="Neutral",
        value=f"{neutral_reviews:,}"
    )

# ------------------------------------------------
# SPACING
# ------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------
# CHART ROW
# ------------------------------------------------

left, middle, right = st.columns([1,1,1])

# ------------------------------------------------
# PIE CHART
# ------------------------------------------------

with left:

    st.markdown("## 📊 Sentiment")

    counts = df["sentiment"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(4,4))

    ax1.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        textprops={'fontsize':10}
    )

    st.pyplot(fig1)

# ------------------------------------------------
# COMPLAINT TYPES
# ------------------------------------------------

with middle:

    st.markdown("## 📈 Complaint Types")

    categories = {

        "Server":620,
        "Accuracy":450,
        "Limits":320,
        "Performance":280,
        "Privacy":40
    }

    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax2.barh(
        list(categories.keys()),
        list(categories.values())
    )

    ax2.tick_params(labelsize=10)

    st.pyplot(fig2)

# ------------------------------------------------
# KEYWORDS
# ------------------------------------------------

with right:

    st.markdown("## ☁️ Top Keywords")

    words = {

        "error":540,
        "chat":500,
        "working":460,
        "voice":420,
        "wrong":360
    }

    fig3, ax3 = plt.subplots(figsize=(5,4))

    ax3.barh(
        list(words.keys()),
        list(words.values())
    )

    ax3.tick_params(labelsize=10)

    st.pyplot(fig3)

# ------------------------------------------------
# BOTTOM ROW
# ------------------------------------------------

left2, right2 = st.columns([1.2,1])

# ------------------------------------------------
# SAMPLE COMPLAINTS
# ------------------------------------------------

with left2:

    st.markdown("## 📝 User Complaints")

    complaints = [

        "Server errors happen too often and responses stop midway.",

        "Voice feature crashes after updates and stops responding.",

        "Usage limits make long conversations frustrating.",

        "AI sometimes gives incorrect answers repeatedly."
    ]

    for text in complaints:

        st.markdown(
            f"""
            <div class="insight-box"
                 style="margin-bottom:10px;
                        border-left:4px solid #ef4444;">

                {text}

            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------------
# AI INSIGHTS
# ------------------------------------------------

with right2:

    st.markdown("## 🤖 AI Insights")

    st.markdown("""
    <div class="insight-box">

    🔴 Server instability is the biggest source of frustration.<br><br>

    🟠 Users strongly dislike waiting limits and restrictions.<br><br>

    🟡 Accuracy problems reduce trust in AI responses.<br><br>

    🔵 Voice and performance issues appear repeatedly in reviews.<br><br>

    🟢 Most users still maintain overall positive sentiment.

    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.caption(
    "Built with Python • Pandas • NLP • Streamlit • Sentiment Analysis"
)