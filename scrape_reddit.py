# scrape_reddit.py
import os
import praw
import requests
from tqdm import tqdm
import prawcore

# CONFIGURATION
REDDIT_CLIENT_ID = 'OYqZFL3_MHiKDbsjQD2HuQ'
REDDIT_CLIENT_SECRET = 'JmOCIcjvGqowK0q4fY2K5ren09GTSA'
USER_AGENT = 'timeline-moderation-script'

# Subreddits to pull from
#APPROPRIATE_SUBREDDITS = ['aww', 'wholesome', 'MadeMeSmile', 'EarthPorn', 'CasualConversation']
#INAPPROPRIATE_SUBREDDITS = ['Instagramreality', 'selfie', 'OnlyFansAdvice', 'RoastMe', 'trashy']
#APPROPRIATE_SUBREDDITS = ['travel','digitalnomad','concerts','festivalfashion','musicfestival','india','pakistan','africanpics','latinopeopletwitter','blurry','lowqualityimages','pets']
#INAPPROPRIATE_SUBREDDITS = ['memes','okbuddyretard','wholesomememes','gymselfie','fitness','entrepreneur','startups','selfimprovement','LinkedInLunatics','cosplay','selfportrait','sneakers','fashionreps','brandnew','AdviceAnimals','PoliticalHumor']
APPROPRIATE_SUBREDDITS = ['']
INAPPROPRIATE_SUBREDDITS = ['bodybuilding']

# How many images per subreddit
NUM_POSTS = 50

# Init Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Download image utility
def download_image(url, folder, filename):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and 'image' in r.headers['Content-Type']:
            with open(os.path.join(folder, filename), 'wb') as f:
                f.write(r.content)
    except Exception as e:
        pass  # skip bad URLs silently

def scrape_and_save(subreddits, folder):
    os.makedirs(folder, exist_ok=True)
    for sub in subreddits:
        print(f"Scraping r/{sub}...")
        try:
            for post in tqdm(reddit.subreddit(sub).hot(limit=NUM_POSTS)):
                if hasattr(post, 'url') and post.url.lower().endswith(('jpg', 'jpeg', 'png')):
                    filename = f"{sub}_{post.id}.jpg"
                    download_image(post.url, folder, filename)
        except prawcore.exceptions.Redirect as e:
            print(f"⚠️  Skipping r/{sub} - subreddit requires authentication or is private")
            continue
        except prawcore.exceptions.NotFound:
            print(f"⚠️  Skipping r/{sub} - subreddit not found")
            continue
        except prawcore.exceptions.Forbidden:
            print(f"⚠️  Skipping r/{sub} - access forbidden")
            continue
        except Exception as e:
            print(f"⚠️  Skipping r/{sub} due to unexpected error: {str(e)}")
            continue

# Start scraping
if __name__ == "__main__":
    scrape_and_save(APPROPRIATE_SUBREDDITS, 'dataset/appropriate')
    scrape_and_save(INAPPROPRIATE_SUBREDDITS, 'dataset/inappropriate')
    print("✅ Scraping complete.")
