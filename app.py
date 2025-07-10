import streamlit as st
import openai
import subprocess
import json
import pandas as pd
import re
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

# Securely load your API key (place this BEFORE calling OpenAI)
openai.api_key = st.secrets["openai_api_key"]


# Text preprocessing
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#", '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", '', text)
    return text.lower().strip()

# Quote tweet scraper
def scrape_quote_tweets(tweet_url):
    command = f'snscrape --jsonl twitter-search \'url:"{tweet_url}"\''
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    quotes = [json.loads(line) for line in result.stdout.splitlines()]
    return quotes

# Topic modeling
def analyze_topics(texts, top_n=10):
    stop_words = stopwords.words('english')
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(texts)

    topic_freq = topic_model.get_topic_info().head(top_n)
    return topic_model, topics, probs, topic_freq

# App UI
st.set_page_config(page_title="Quote Tweet Analyzer", layout="wide")

st.title("🧠 Twitter Quote Tweet Topic Analyzer")

tweet_url = st.text_input("Paste the Tweet URL here")

if tweet_url and st.button("Analyze Quote Tweets"):
    with st.spinner("Scraping quote tweets..."):
        quotes = scrape_quote_tweets(tweet_url)
        cleaned_texts = list(set(clean_text(q["content"]) for q in quotes if q.get("content")))
        st.success(f"Scraped {len(cleaned_texts)} unique quote tweets.")

    if cleaned_texts:
        with st.spinner("Running topic modeling..."):
            topic_model, topics, probs, topic_freq = analyze_topics(cleaned_texts)

        st.subheader("Top Topics Identified")
        st.dataframe(topic_freq)

        # Download CSV
        df = pd.DataFrame({"text": cleaned_texts, "topic": topics})
        st.download_button("Download CSV", df.to_csv(index=False), "quote_tweets_topics.csv", "text/csv")

        # Visualization
        st.subheader("📊 Topic Distribution")
        fig = px.bar(topic_freq.head(10), x='Topic', y='Count', text='Name')
        st.plotly_chart(fig)

        if st.checkbox("Show All Analyzed Tweets"):
            st.dataframe(df)

        if st.checkbox("Show Interactive Topic Visualization"):
            st.plotly_chart(topic_model.visualize_barchart(top_n_topics=10))
