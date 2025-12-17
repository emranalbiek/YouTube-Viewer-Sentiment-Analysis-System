import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
from googleapiclient.discovery import build

# Page configuration
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric label {
        color: #9ca3af !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stAlert {
        background-color: #1e2130;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK data (download once)
@st.cache_resource
def download_nltk_data():
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Preprocessing function
@st.cache_data
def preprocess_comment(comment):
    """Preprocess a single comment"""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        return comment
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return comment

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load trained model and vectorizer"""
    try:
        with open('artifacts/lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('artifacts/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found in 'artifacts/' folder!")
        st.info("Make sure these files exist:\n- artifacts/lightgbm_model.pkl\n- artifacts/tfidf_vectorizer.pkl")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Fetch YouTube comments
@st.cache_data(ttl=3600)
def fetch_youtube_comments(video_id, api_key, max_comments=20000):
    """Fetch comments from YouTube API"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        page_token = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=page_token,
                textFormat='plainText'
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textOriginal'],
                    'timestamp': comment['publishedAt'],
                    'author': comment['authorDisplayName'],
                    'authorId': comment.get('authorChannelId', {}).get('value', 'Unknown')
                })
            
            page_token = response.get('nextPageToken')
            progress = min(len(comments) / max_comments, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Fetched {len(comments)} comments...")
            
            if not page_token:
                break
        
        progress_bar.empty()
        status_text.empty()
        return comments[:max_comments]
        
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return []

# Extract video ID from URL
def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Predict sentiments
def predict_sentiments(comments, model, vectorizer):
    """Predict sentiments for comments"""
    try:
        texts = [c['text'] for c in comments]
        preprocessed = [preprocess_comment(text) for text in texts]
        
        with st.spinner("🤖 Analyzing sentiments..."):
            transformed = vectorizer.transform(preprocessed)
            predictions = model.predict(transformed.toarray())
        
        # Reverse mapping: 2→-1 (Neg), 0→0 (Neu), 1→1 (Pos)
        mapping = {2: -1, 0: 0, 1: 1}
        predictions = [mapping[p] for p in predictions]
        
        for comment, pred in zip(comments, predictions):
            comment['sentiment'] = pred
        
        return comments
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return []

# Generate pie chart
def generate_pie_chart(sentiment_counts):
    """Generate sentiment distribution pie chart"""
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [
        sentiment_counts.get(1, 0),
        sentiment_counts.get(0, 0),
        sentiment_counts.get(-1, 0)
    ]
    colors = ['#36A2EB', '#C9CBCF', '#FF6384']
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0e1117')
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, textprops={'color': 'white', 'fontsize': 14})
    ax.axis('equal')
    plt.title('Sentiment Distribution', color='white', fontsize=18, pad=20)
    return fig

# Generate word cloud
def generate_wordcloud(comments):
    """Generate word cloud from comments"""
    text = ' '.join([preprocess_comment(c['text']) for c in comments])
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='#0e1117',
        colormap='Blues',
        stopwords=set(stopwords.words('english')),
        collocations=False,
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title('Comment Word Cloud', color='white', fontsize=18, pad=20)
    return fig

# Generate trend graph
def generate_trend_graph(comments):
    """Generate sentiment trend over time"""
    df = pd.DataFrame(comments)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample monthly
    monthly = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
    
    for val in [-1, 0, 1]:
        if val not in monthly.columns:
            monthly[val] = 0
    
    monthly = monthly[[-1, 0, 1]]
    totals = monthly.sum(axis=1)
    percentages = (monthly.T / totals).T * 100
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    for val in [-1, 0, 1]:
        ax.plot(percentages.index, percentages[val], 
                marker='o', linestyle='-', color=colors[val], 
                label=labels[val], linewidth=2, markersize=8)
    
    ax.set_facecolor('#0e1117')
    ax.set_title('Sentiment Trend Over Time', color='white', fontsize=18, pad=20)
    ax.set_xlabel('Month', color='white', fontsize=14)
    ax.set_ylabel('Percentage (%)', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.legend(facecolor='#1e2130', edgecolor='white', labelcolor='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Main app
def main():
    # Header
    st.title("🎥 YouTube Comment Sentiment Analyzer")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="Get your API key from Google Cloud Console"
        )
        
        max_comments = st.slider(
            "Max Comments",
            min_value=100,
            max_value=20000,
            value=1000,
            step=100
        )
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Enter your YouTube API key
        2. Paste a YouTube video URL
        3. Click 'Analyze' button
        4. View results and insights
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Get API Key")
        st.markdown("[Google Cloud Console](https://console.cloud.google.com/)")
        
        st.markdown("### Connect with me")
        st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <div style="display: flex; gap: 20px; font-size: 28px;">
                <a href="http://linkedin.com/in/emranalbeik" target="_blank"><i class="fab fa-linkedin"></i></a>
                <a href="https://github.com/RedDragon30" target="_blank"><i class="fab fa-github"></i></a>
                <a href="https://emranalbeik.odoo.com/" target="_blank"><i class="fas fa-globe"></i></a>
                <a href="mailto:emranalbiek@gmail.com"><i class="fas fa-envelope"></i></a>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    video_input = st.text_input(
        "🔗 Enter YouTube Video URL or ID",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID"
    )
    
    if st.button("🚀 Analyze Comments", type="primary", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please enter your YouTube API key in the sidebar!")
            return
        
        if not video_input:
            st.error("⚠️ Please enter a YouTube video URL or ID!")
            return
        
        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("❌ Invalid YouTube URL or ID!")
            return
        
        # Display video info
        st.markdown(f"### 📺 Video ID: `{video_id}`")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        # Load model
        model, vectorizer = load_model()
        
        # Fetch comments
        with st.spinner(f"📥 Fetching up to {max_comments} comments..."):
            comments = fetch_youtube_comments(video_id, api_key, max_comments)
        
        if not comments:
            st.error("❌ No comments found or API error occurred!")
            return
        
        st.success(f"✅ Fetched {len(comments)} comments successfully!")
        
        # Predict sentiments
        comments = predict_sentiments(comments, model, vectorizer)
        
        if not comments:
            return
        
        # Calculate metrics
        sentiment_counts = {-1: 0, 0: 0, 1: 0}
        for c in comments:
            sentiment_counts[c['sentiment']] += 1
        
        total = len(comments)
        unique_authors = len(set(c['authorId'] for c in comments))
        avg_length = sum(len(c['text'].split()) for c in comments) / total
        avg_sentiment = sum(c['sentiment'] for c in comments) / total
        normalized_score = ((avg_sentiment + 1) / 2) * 10
        
        # Display metrics
        st.markdown("---")
        st.markdown("## 📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments", f"{total:,}")
        with col2:
            st.metric("Unique Commenters", f"{unique_authors:,}")
        with col3:
            st.metric("Avg Comment Length", f"{avg_length:.1f} words")
        with col4:
            st.metric("Sentiment Score", f"{normalized_score:.1f}/10")
        
        # Sentiment breakdown
        st.markdown("---")
        st.markdown("## 🎯 Sentiment Breakdown")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "😊 Positive",
                f"{sentiment_counts[1]:,}",
                f"{(sentiment_counts[1]/total*100):.1f}%"
            )
        with col2:
            st.metric(
                "😐 Neutral",
                f"{sentiment_counts[0]:,}",
                f"{(sentiment_counts[0]/total*100):.1f}%"
            )
        with col3:
            st.metric(
                "😞 Negative",
                f"{sentiment_counts[-1]:,}",
                f"{(sentiment_counts[-1]/total*100):.1f}%"
            )
        
        # Visualizations
        st.markdown("---")
        st.markdown("## 📈 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Pie Chart", "☁️ Word Cloud", "📉 Trend", "💬 Comments"])
        
        with tab1:
            fig = generate_pie_chart(sentiment_counts)
            st.pyplot(fig)
        
        with tab2:
            fig = generate_wordcloud(comments)
            st.pyplot(fig)
        
        with tab3:
            fig = generate_trend_graph(comments)
            st.pyplot(fig)
        
        with tab4:
            st.markdown("### Recent Comments")
            sentiment_filter = st.selectbox(
                "Filter by sentiment:",
                ["All", "Positive", "Neutral", "Negative"]
            )
            
            sentiment_map = {"All": None, "Positive": 1, "Neutral": 0, "Negative": -1}
            filtered = [c for c in comments if sentiment_filter == "All" or c['sentiment'] == sentiment_map[sentiment_filter]]
            
            for i, c in enumerate(filtered[:50], 1):
                sentiment_emoji = {1: "😊", 0: "😐", -1: "😞"}
                with st.expander(f"{sentiment_emoji[c['sentiment']]} {c['author']} - {c['timestamp'][:10]}"):
                    st.write(c['text'])

if __name__ == "__main__":
    main()
