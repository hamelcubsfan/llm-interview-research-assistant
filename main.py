# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit
import streamlit as st

# Twitter
import tweepy

# Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# YouTube
from langchain.document_loaders import YoutubeLoader

# Get your API keys from Streamlit secrets
TWITTER_API_KEY = st.secrets["general"]["TWITTER_API_KEY"]
TWITTER_API_SECRET = st.secrets["general"]["TWITTER_API_SECRET"]
TWITTER_ACCESS_TOKEN = st.secrets["general"]["TWITTER_ACCESS_TOKEN"]
TWITTER_ACCESS_TOKEN_SECRET = st.secrets["general"]["TWITTER_ACCESS_TOKEN_SECRET"]
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]

# Load up your LLM
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key, max_tokens=2000, model_name='gpt-3.5-turbo')
    return llm

# A function that will be called only if the environment's openai_api_key isn't set
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

# We'll query 80 tweets because we end up filtering out a bunch
def get_original_tweets(screen_name, tweets_to_pull=80, tweets_to_return=80):
    st.write("Getting Tweets...")
    # Tweepy set up
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    # Holder for the tweets you'll find
    tweets = []
    
    # Go and pull the tweets
    tweepy_results = tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended',
                                   exclude_replies=True).items(tweets_to_pull)
    
    # Run through tweets and remove retweets and quote tweets so we can only look at a user's raw emotions
    for status in tweepy_results:
        if hasattr(status, 'retweeted_status') or hasattr(status, 'quoted_status'):
            # Skip if it's a retweet or quote tweet
            continue
        else:
            tweets.append({'full_text': status.full_text, 'likes': status.favorite_count})

    
    # Sort the tweets by number of likes. This will help us short_list the top ones later
    sorted_tweets = sorted(tweets, key=lambda x: x['likes'], reverse=True)

    # Get the text and drop the like count from the dictionary
    full_text = [x['full_text'] for x in sorted_tweets][:tweets_to_return]
    
    # Convert the list of tweets into a string of tweets we can use in the prompt later
    users_tweets = "\n\n".join(full_text)
            
    return users_tweets

# Function for getting YouTube comments
def get_youtube_comments(video_id, comments_to_pull=20):
    st.write("Getting YouTube Comments...")
    comments = YoutubeLoader().get_youtube_comments(video_id, comments_to_pull)
    return comments

def main():
    llm = load_LLM(OPENAI_API_KEY)

    username = st.text_input(label="Twitter Username (no @ symbol)", placeholder="Ex: jack", key="username")
    video_id = st.text_input(label="YouTube Video ID", placeholder="Ex: dQw4w9WgXcQ", key="video_id")

    if username and video_id:
        st.write("Loading Tweets and YouTube Comments...")
        raw_emotions_twitter = get_original_tweets(username)
        raw_emotions_youtube = get_youtube_comments(video_id)
        raw_emotions = raw_emotions_twitter + " " + raw_emotions_youtube

        splitter = RecursiveTokenSplitter(max_tokens=4096)
        emotion_splits = splitter.split_text(raw_emotions)

        combined_responses = []

        for split in emotion_splits:
            prompt = f"{split}"
            response = llm(prompt)
            combined_responses.append(response['message']['content'])
        
        st.write(" ".join(combined_responses))

    else:
        st.write("Please enter both a Twitter username and a YouTube Video ID.")

if __name__ == "__main__":
    main()

