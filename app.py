from flask import Flask, render_template, jsonify, request, redirect, url_for
from openai import OpenAI
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

client = OpenAI(api_key='sk-proj-dBBXHTwmjWGpqIwlPq9qT3BlbkFJ3PTTr5nn12T5eVHQyXnH')

app = Flask(__name__)

feed_urls = {
    "general": [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
        "https://www.theguardian.com/world/rss",
        "http://rss.cnn.com/rss/edition_world.rss",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "http://feeds.foxnews.com/foxnews/latest",
        "http://feeds.washingtonpost.com/rss/national",
        "https://www.npr.org/rss/rss.php?id=1001",
        "https://www.latimes.com/local/rss2.0.xml",
    ],
    "finance": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "http://feeds.washingtonpost.com/rss/business",
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "http://feeds.reuters.com/reuters/businessNews",
        "http://rss.cnn.com/rss/money_news_international.rss",
        "https://www.ft.com/?format=rss",
        "http://feeds.skynews.com/feeds/rss/business.xml",
    ],
    "technology": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "http://rss.cnn.com/rss/edition_technology.rss",
        "http://feeds.foxnews.com/foxnews/tech",
        "http://feeds.washingtonpost.com/rss/business/technology",
        "https://www.npr.org/rss/rss.php?id=1019",
        "https://www.latimes.com/business/technology/rss2.0.xml",
    ],
    "health": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
        "http://feeds.bbci.co.uk/news/health/rss.xml",
        "https://www.cnbc.com/id/10000108/device/rss/rss.html",
        "http://feeds.foxnews.com/foxnews/health",
        "http://feeds.washingtonpost.com/rss/national/health-science",
        "https://www.npr.org/rss/rss.php?id=1003",
        "https://www.latimes.com/health/rss2.0.xml",
    ],
    "sports": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
        "http://feeds.bbci.co.uk/sport/rss.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "http://rss.cnn.com/rss/edition_sport.rss",
        "http://feeds.foxnews.com/foxnews/sports",
        "http://feeds.washingtonpost.com/rss/sports",
        "https://www.npr.org/rss/rss.php?id=1055",
        "https://www.latimes.com/sports/rss2.0.xml",
    ],
    "entertainment": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
        "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
        "https://www.cnbc.com/id/10000739/device/rss/rss.html",
        "http://rss.cnn.com/rss/edition_entertainment.rss",
        "http://feeds.foxnews.com/foxnews/entertainment",
        "http://feeds.washingtonpost.com/rss/entertainment",
        "https://www.npr.org/rss/rss.php?id=1045",
        "https://www.latimes.com/entertainment-arts/rss2.0.xml",
    ]

}

def fetch_top_headlines(category):
    articles = []
    for feed_url in feed_urls.get(category, []):
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            print(f"Error fetching feed {feed_url}: {feed.bozo_exception}")
            continue
        for entry in feed.entries:
            title = entry.get('title', 'No title')
            description = entry.get('description', title)
            link = entry.get('link', '')
            articles.append({"title": title, "description": description, "link": link})
    return articles

def score_articles(articles):
    keywords = [
        'breaking', 'urgent', 'exclusive', 'important', 'update', 'alert', 
        'major', 'critical', 'significant', 'essential', 'notable', 'top story'
    ]
    keyword_counter = Counter(keywords)

    def get_score(article):
        title = article['title'].lower()
        description = article['description'].lower()
        word_count = len(re.findall(r'\w+', title + ' ' + description))
        keyword_score = sum(keyword_counter[word] for word in title.split() if word in keyword_counter)
        keyword_score += sum(keyword_counter[word] for word in description.split() if word in keyword_counter)
        return keyword_score + word_count

    return sorted(articles, key=get_score, reverse=True)

def summarize_article(article):
    article_text = f"Title: {article['title']}\n\n{article['description']}\n\nSource: {article['link']}"
    prompt = (
        "You are an AI language model designed to summarize newspaper articles. Your goal is to provide a concise, accurate, "
        "and informative summary of the given article. Follow these guidelines:\n"
        "Main Points and Key Details: Highlight any important details, including names, dates, statistics, and quotes. Identify and bulletlist with numbers the main points covered in the article.\n"
        "Concise: Do NOT write in your response anything other than bulletlist of main points and key details."
        "DO NOT WRITE ANYTHING OTHER THAN THE BULLETLIST"
        "DO NOT INCLUDE SOURCE IN YOUR RESPONSE. I WILL DO IT MYSELF. DO NOT INCLUDE ANYTHING OTHER THAN THE BULLET LIST OF THE SUMMARY"
        "Objective Tone: Maintain an objective and neutral tone throughout the summary.\n"
        "Length: Aim for a summary length of around 100 - 150 words.\n"
        "Relevance: Ensure that the summary covers all relevant aspects of the article without unnecessary information.\n"
        f"{article_text}\n\nSummary:"
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes the newspapers for the user."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    summary = response.choices[0].message.content.strip()
    bullet_points = summary.split('\n')
    formatted_summary = ''.join(f"<li>{point}</li>" for point in bullet_points if point.strip())
    return {
        "title": article["title"],
        "description": f"<ul>{formatted_summary}</ul>",
        "link": article["link"]
    }

def filter_similar_articles(articles, threshold=0.7):
    descriptions = [article['description'] for article in articles]
    vectorizer = TfidfVectorizer().fit_transform(descriptions)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    unique_articles = []
    for idx, article in enumerate(articles):
        if all(cosine_matrix[idx][i] < threshold for i in range(len(articles)) if i != idx):
            unique_articles.append(article)
    
    return unique_articles

def curate_news():
    curated_news = {}
    for category in feed_urls.keys():
        articles = fetch_top_headlines(category)
        scored_articles = score_articles(articles)
        diverse_articles = filter_similar_articles(scored_articles)
        summaries = []
        for article in diverse_articles[:8]:  # Summarize only top 5 diverse articles
            summary = summarize_article(article)
            summaries.append(summary)
        curated_news[category] = summaries
    return curated_news

curated_news = curate_news()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/category/<category>')
def category_page(category):
    articles = get_articles(category)  # Assume this function fetches articles based on category
    return render_template('index.html', category=category, articles=articles)

@app.route('/search')
def search():
    query = request.args.get('q')
    search_results = search_articles(query)  # Assume this function searches articles based on query
    return render_template('search_results.html', query=query, articles=search_results)

if __name__ == '__main__':
    app.run(debug=True)
