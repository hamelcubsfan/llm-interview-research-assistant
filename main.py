# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

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
    llm = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key, max_tokens=2000, model_name='gpt-4')
    return llm

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

def get_original_tweets(screen_name, tweets_to_pull=80, tweets_to_return=80):
    st.write("Getting Tweets...")
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    tweets = []
    tweepy_results = tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended',
                                   exclude_replies=True).items(tweets_to_pull)
    for status in tweepy_results:
        if hasattr(status, 'retweeted_status') or hasattr(status, 'quoted_status'):
            continue
        else:
            tweets.append({'full_text': status.full_text, 'likes': status.favorite_count})
    sorted_tweets = sorted(tweets, key=lambda x: x['likes'], reverse=True)
    full_text = [x['full_text'] for x in sorted_tweets][:tweets_to_return]
    users_tweets = "\n\n".join(full_text)
    return users_tweets

def pull_from_website(url):
    st.write("Getting webpages...")
    try:
        response = requests.get(url)
    except:
        print ("Whoops, error")
        return
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    text = md(text)
    return text

def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

def split_text(user_information):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
    docs = text_splitter.create_documents([user_information])
    return docs

response_types = {
    'Email' : """
        "Your task is to compose an email to {recipient_name}, summarizing the main points and takeaways of your conversation with {source_name}. Use a professional and concise tone."
    """,
    'Presentation' : """
        "Your task is to compose a presentation script summarizing the main points and takeaways of your conversation with {source_name}. Make sure it's engaging and clear for your audience."
    """,
    'Report' : """
        "Your task is to write a report summarizing the main points and takeaways of your conversation with {source_name}. Use a formal and detailed tone."
    """
}

def ask_for_input():
    input_type = st.selectbox(label="What type of information do you want to summarize?", 
                              options=['Twitter', 'Webpage', 'YouTube Video'], 
                              key="input_type_select")
    if input_type == 'Twitter':
        screen_name = st.text_input(label="Twitter username",  placeholder="Ex: elonmusk", key="screen_name_input")
        original_information = get_original_tweets(screen_name) if screen_name else ''
    elif input_type == 'Webpage':
        url = st.text_input(label="Webpage URL",  placeholder="Ex: https://...", key="url_input")
        original_information = pull_from_website(url) if url else ''
    elif input_type == 'YouTube Video':
        url = st.text_input(label="YouTube Video URL",  placeholder="Ex: https://...", key="youtube_url_input")
        original_information = get_video_transcripts(url) if url else ''
    return input_type, original_information

def ask_for_output():
    output_type = st.selectbox(label="In what format do you want the summary?", 
                               options=['Email', 'Presentation', 'Report'], 
                               key="output_type_select")
    recipient_name = st.text_input(label="Recipient's name",  placeholder="Ex: Bob", key="recipient_name_input") if output_type == 'Email' else ''
    source_name = st.text_input(label="Source's name",  placeholder="Ex: Elon Musk", key="source_name_input")
    return output_type, recipient_name, source_name

def main():
    openai_api_key = get_openai_api_key() or OPENAI_API_KEY
    if not openai_api_key:
        st.warning("Please input OpenAI API Key")
        return
    llm = load_LLM(openai_api_key)
    input_type, original_information = ask_for_input()
    if original_information:
        documents = split_text(original_information)
        output_type, recipient_name, source_name = ask_for_output()
        if source_name:
            summarize_chain = load_summarize_chain(llm, documents)
            prompt_template = PromptTemplate(response_types[output_type])
            context = {'recipient_name': recipient_name, 'source_name': source_name}
            results = summarize_chain(prompt_template, context)
            st.markdown(results)

if __name__ == "__main__":
    main()

