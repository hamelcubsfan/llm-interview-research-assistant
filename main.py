# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

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
# !pip install youtube-transcript-api

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()

# Get your API keys set
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', 'YourAPIKeyIfNotSet')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', 'YourAPIKeyIfNotSet')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', 'YourAPIKeyIfNotSet')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'YourAPIKeyIfNotSet')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YouAPIKeyIfNotSet')

# Load up your LLM
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key, max_tokens=2000, model_name='gpt-4')
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
    users_tweets = "\n\n---\n\n".join(full_text)
    
    return users_tweets

# Ask the user for a Twitter handle
st.title('Twitter Persona Creation')
screen_name = st.text_input('Enter a Twitter handle: @', '')

# If it's set, let's go!
if screen_name:
    # Get the Tweets
    users_tweets = get_original_tweets(screen_name, tweets_to_pull=80, tweets_to_return=50)
    
    # Format the prompt
    prompt = PromptTemplate(persona_formation).render(user_input=users_tweets)
    
    # Make sure the LLM is loaded
    if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
        OPENAI_API_KEY = get_openai_api_key()
    llm = load_LLM(openai_api_key=OPENAI_API_KEY)
    
    # Get the model's response
    model_response = llm.respond(prompt=prompt)
    
    # Output the model's response
    st.write(model_response)
