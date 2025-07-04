import praw
import pandas as pd
from config.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from utils.logger import log_info

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def get_reddit_posts(subreddit_name, limit=10):
    log_info(f"Scraping Reddit: {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            'title': post.title,
            'text': post.selftext,
            'score': post.score,
            'created': post.created_utc
        })
    df = pd.DataFrame(posts)
    df.to_csv(f"data/osint/reddit_{subreddit_name}.csv", index=False)
    log_info(f"Saved Reddit data for {subreddit_name}")
