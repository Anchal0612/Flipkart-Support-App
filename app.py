# 📦 Import all useful tools
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# 🖼️ Page ka title and layout set karte hain
st.set_page_config(page_title="Flipkart Support Dashboard", layout="wide")

# 🧾 Load CSV data function
@st.cache_data
def load_data():
    # Yeh function humara data load karega
    data = pd.read_csv("Customer_support_data.csv")  # make sure this file is in same folder
    return data

# 📂 Data ko load kar lo
df = load_data()

# 🧠 App ka title
st.title("📞 Flipkart Customer Support Analysis")

# ➕ Tabs banate hain: Overview | Sentiment | Raw Data
tab1, tab2, tab3 = st.tabs(["📊 Overview", "💬 Sentiment", "📄 Raw Data"])

# ------------------------------------------
# 📊 Tab 1: CSAT Score Overview
# ------------------------------------------
with tab1:
    st.header("Agent Performance - CSAT Score Analysis")

    # 🚥 Shift-wise average CSAT
    st.subheader("1️⃣ Average CSAT by Agent Shift")
    shift_csat = df.groupby("Agent Shift")["CSAT Score"].mean()
    st.bar_chart(shift_csat)

    # ⏳ Tenure Bucket-wise average CSAT
    st.subheader("2️⃣ Average CSAT by Agent Experience")
    tenure_csat = df.groupby("Tenure Bucket")["CSAT Score"].mean()
    st.bar_chart(tenure_csat)

    # 📦 Complaint Category-wise average CSAT
    st.subheader("3️⃣ Average CSAT by Complaint Category")
    category_csat = df.groupby("category")["CSAT Score"].mean()
    st.bar_chart(category_csat)

# ------------------------------------------
# 💬 Tab 2: Sentiment Analysis of Remarks
# ------------------------------------------
with tab2:
    st.header("What Customers Are Saying (Sentiment Analysis)")

    # 💬 Filter only rows jisme remarks ho
    df_cleaned = df[['Customer Remarks', 'CSAT Score']].dropna()

    # 🧠 Sentiment score lagao using TextBlob
    df_cleaned['Sentiment'] = df_cleaned['Customer Remarks'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    # 🔠 Label banao: Positive / Negative / Neutral
    def get_sentiment_label(score):
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    df_cleaned['Sentiment_Label'] = df_cleaned['Sentiment'].apply(get_sentiment_label)

    # 📊 Count plot
    st.subheader("Sentiment Count (Positive / Negative / Neutral)")
    st.bar_chart(df_cleaned['Sentiment_Label'].value_counts())

    # ☁️ WordCloud bana ke dikhana
    st.subheader("Most Common Words Used in Remarks")
    all_text = " ".join(df_cleaned['Customer Remarks'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# ------------------------------------------
# 📄 Tab 3: Raw Data Table
# ------------------------------------------
with tab3:
    st.subheader("Full Dataset View")
    st.dataframe(df)

