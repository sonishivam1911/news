import os
import asyncio
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from nltk.corpus import stopwords
import string

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Get API keys from environment variables
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
CURRENTSAPI_KEY = os.getenv("CURRENTSAPI_KEY")
GUARDIANAPI_KEY = os.getenv("GUARDIANAPI_KEY")

# Initialize NLP pipelines and stopwords
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
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

# Compute relevance scores using TF-IDF + cosine similarity (async)
async def compute_relevance_score(query, article):
    content = article.get("description") or article.get("content") or ""
    if not content:
        return None

    preprocessed_query = preprocess_text(query)
    preprocessed_content = preprocess_text(content)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_query, preprocessed_content])

    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return {
        "source": article.get("source", {}).get("name", ""),
        "url": article.get("url", ""),
        "content": content,
        "relevance_score": score,
    }

# Score all articles concurrently (async)
async def score_articles_concurrently(query, articles):
    tasks = [compute_relevance_score(query, article) for article in articles]
    scored_articles = await asyncio.gather(*tasks)
    
    # Filter out None results (articles without content)
    scored_articles = [article for article in scored_articles if article is not None]

    # Sort articles by relevance score in descending order
    scored_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
    print(scored_articles)
    
    # Return top 20 articles with score >= 0.7 (to ensure enough data for both sentiments)
    return [article for article in scored_articles if article["relevance_score"] >= 0.2][:20]

# Analyze sentiments of articles (batch processing) and ensure 10 positive and 10 negative articles
def analyze_sentiments(articles):
    pro_articles, con_articles = [], []
    
    contents_with_metadata = [
        {"content": a["content"], "source": a["source"], "url": a["url"]} for a in articles if a.get("content")
    ]
    
    contents = [item["content"] for item in contents_with_metadata]
    
    sentiments = sentiment_analyzer(contents, batch_size=8)
    
    for item, sentiment in zip(contents_with_metadata, sentiments):
        if sentiment["label"] == "POSITIVE" and len(pro_articles) < 10:
            pro_articles.append(item)
        elif sentiment["label"] == "NEGATIVE" and len(con_articles) < 10:
            con_articles.append(item)

    return pro_articles, con_articles

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

    # Score articles asynchronously using TF-IDF and cosine similarity
    scored_articles = loop.run_until_complete(score_articles_concurrently(query, all_articles))
    
    # Perform sentiment analysis on top relevant articles and ensure 10 positive/negative articles
    pro_articles, con_articles = analyze_sentiments(scored_articles)

    return {
        "pro_articles": pro_articles,
        "con_articles": con_articles,
        "positive_count": len(pro_articles),
        "negative_count": len(con_articles),
        "total_count": len(scored_articles),
        "relevant_articles": scored_articles,
    }