import os
import asyncio
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  # BM25 library
from transformers import pipeline
from nltk.corpus import stopwords
import string
from keybert import KeyBERT  # For keyword extraction

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Get API keys from environment variables
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
CURRENTSAPI_KEY = os.getenv("CURRENTSAPI_KEY")
GUARDIANAPI_KEY = os.getenv("GUARDIANAPI_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize NLP pipelines and stopwords
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Improved summarization model
keyword_extractor = KeyBERT()  # Keyword extraction model
STOP_WORDS = set(stopwords.words("english"))

# Preprocess text: tokenize, remove punctuation/stopwords
def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

# Fetch news using aiohttp (asynchronous)
async def fetch_news_async(url, headers=None):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Failed to fetch data: {response.status}"}

# Fetch news from all APIs asynchronously
async def fetch_all_news(query):
    urls = [
        f"https://newsapi.org/v2/everything?q={query}&pageSize=20&apiKey={NEWSAPI_KEY}",
        f"https://content.guardianapis.com/search?q={query}&page-size=20&api-key={GUARDIANAPI_KEY}",
    ]
    headers_currents = {"Authorization": CURRENTSAPI_KEY}
    urls.append(("https://api.currentsapi.services/v1/latest-news", headers_currents))

    tasks = [
        fetch_news_async(url) if isinstance(url, str) else fetch_news_async(url[0], url[1]) for url in urls
    ]
    return await asyncio.gather(*tasks)

# Compute relevance scores using BM25 (async)
async def compute_bm25_scores(query, articles):
    preprocessed_query = preprocess_text(query)
    tokenized_articles = [
        preprocess_text(article.get("description", "") + " " + article.get("content", "")).split()
        for article in articles if article.get("description") or article.get("content")
    ]
    
    bm25 = BM25Okapi(tokenized_articles)
    scores = bm25.get_scores(preprocessed_query.split())

    scored_articles = []
    for article, score in zip(articles, scores):
        scored_articles.append({
            "source": article.get("source", {}).get("name", ""),
            "url": article.get("url", ""),
            "content": article.get("description", "") + " " + article.get("content", ""),
            "relevance_score": score,
        })

    scored_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored_articles[:20]  # Return top 20 articles

# Analyze sentiments of articles (batch processing)
def analyze_sentiments(articles):
    pro_articles, con_articles = [], []

    contents_with_metadata = [
        {"content": a["content"], "source": a["source"], "url": a["url"]} for a in articles if a.get("content")
    ]

    contents = [item["content"] for item in contents_with_metadata]

    sentiments = sentiment_analyzer(contents, batch_size=8)

    for item, sentiment in zip(contents_with_metadata, sentiments):
        if sentiment["label"] == "POSITIVE":
            pro_articles.append(item)
        elif sentiment["label"] == "NEGATIVE":
            con_articles.append(item)

    return pro_articles[:10], con_articles[:10]

# Summarize all positive or negative articles into a single summary
def summarize_group(articles):
    combined_text = " ".join([article["content"] for article in articles])
    
    try:
        summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Error summarizing group: {e}"

# Extract keywords from an article's content
def extract_keywords(content):
    keywords = keyword_extractor.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)
    return [kw[0] for kw in keywords]

# Main processing function with async scoring and sentiment analysis
def process_news(query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    newsapi_data, guardian_data, currents_data = loop.run_until_complete(fetch_all_news(query))

    all_articles = []
    
    if isinstance(newsapi_data.get("articles"), list):
        all_articles.extend(newsapi_data["articles"])
    
    if isinstance(guardian_data.get("response", {}).get("results"), list):
        all_articles.extend(guardian_data["response"]["results"])
    
    if isinstance(currents_data.get("news"), list):
        all_articles.extend(currents_data["news"])

    # Score articles asynchronously using BM25
    scored_articles_bm25 = loop.run_until_complete(compute_bm25_scores(query, all_articles))

    # Perform sentiment analysis on top relevant articles
    pro_articles, con_articles = analyze_sentiments(scored_articles_bm25)

    # Generate overall summaries for positive/negative articles
    positive_summary = summarize_group(pro_articles)
    negative_summary = summarize_group(con_articles)

    # Extract keywords for each article
    for article in pro_articles + con_articles:
        article["keywords"] = extract_keywords(article["content"])

    return {
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "pro_articles": pro_articles,
        "con_articles": con_articles,
        "positive_count": len(pro_articles),
        "negative_count": len(con_articles),
        "total_count": len(scored_articles_bm25),
        "relevant_articles": scored_articles_bm25,
    }