from flask import Flask, render_template, jsonify, request
from openai import OpenAI
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import os  # Import the os module

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Fetch the API key from environment variable

app = Flask(__name__)

feed_urls = {
    "general": [
        "http://feeds.bbci.co.uk/news/rss.xml"
    ],
    "finance": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    ],
    "technology": [
        "http://feeds.bbci.co.uk/news/technology/rss.xml"
    ],
    "health": [
        "http://feeds.bbci.co.uk/news/health/rss.xml"
    ],
    "sports": [
        "http://feeds.bbci.co.uk/sport/rss.xml"
    ],
    "entertainment": [
        "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"
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
    keywords = ['breaking', 'urgent', 'exclusive', 'important', 'update', 'alert', 
                'major', 'critical', 'significant', 'essential', 'notable', 'top story']
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
        "Main Points and Key Details: Highlight any important details, including names, dates, statistics, and quotes. Identify and bullet list with numbers the main points covered in the article.\n"
        "Concise: Do NOT write in your response anything other than bullet list of main points and key details.\n"
        "DO NOT WRITE ANYTHING OTHER THAN THE BULLET LIST\n"
        "DO NOT INCLUDE SOURCE IN YOUR RESPONSE. I WILL DO IT MYSELF. DO NOT INCLUDE ANYTHING OTHER THAN THE BULLET LIST OF THE SUMMARY\n"
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
        for article in diverse_articles[:8]:  # Summarize only top 8 diverse articles
            summary = summarize_article(article)
            summaries.append(summary)
        curated_news[category] = summaries
    return curated_news

curated_news = curate_news()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/category/<category>')
def show_category(category):
    if category not in curated_news:
        return "Category not found", 404
    return render_template('index.html', category=category, articles=curated_news[category])

@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = []
    if query:
        for category, articles in curated_news.items():
            for article in articles:
                if query.lower() in article['title'].lower() or query.lower() in article['description'].lower():
                    results.append(article)
    return render_template('index.html', category='Search Results', articles=results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
