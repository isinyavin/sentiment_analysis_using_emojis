import praw
import logging
from datetime import datetime
import pandas as pd
import emoji
import time
from prawcore.exceptions import RequestException, ResponseException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

client_id = 'cd0ceYs7urf4HRJo4R-4NA'
client_secret = 'eNrNXBmA5ZTXl3PvIC3IhFWxY0VZ-g'
user_agent = 'emoji-sentiment-analysis-script by /u/russian-rabbit'

try:

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)


    logger.info("Authenticated as: %s", reddit.user.me())

    subreddit = reddit.subreddit('travel')

    data = []

    for post in subreddit.top(limit=100000):
        while True:
            try:
                post.comments.replace_more(limit=20)
                for comment in post.comments.list():
                    if emoji.emoji_count(comment.body) > 0:
                        comment_data = {
                            'post_id': post.id,
                            'comment_body': comment.body,
                            'comment_created_utc': datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        data.append(comment_data)
                        print(comment_data['comment_body'])

                df = pd.DataFrame(data)

                df.to_csv('reddit_comments_with_emojis22.csv', index=False)
                logger.info(f"Data saved to reddit_comments_with_emojis10.csv after processing post {post.id}")

                break  
            except (RequestException, ResponseException) as e:
                if 'RATELIMIT' in str(e):
                    wait_time = int(str(e).split(' ')[-2])
                    logger.info(f"Rate limit hit. Waiting for {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    logger.error(f"RequestException or ResponseException: {e}")
                    time.sleep(10) 

    logger.info("Scraping completed and data saved to reddit_comments_with_emojis9.csv")

except praw.exceptions.PRAWException as e:
    logger.error("PRAWException: %s", e)
except Exception as e:
    logger.error("Exception: %s", e)