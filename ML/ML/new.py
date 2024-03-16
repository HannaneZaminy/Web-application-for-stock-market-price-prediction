from stocknews import StockNews
import streamlit as st

st.set_page_config(page_title="NEWS" , page_icon=":bar_chart:" ,layout="wide")
symbol = st.sidebar.selectbox('Select The Symbol : ', ['BTC-USD', 'AMZN', 'ETH-USD', 'TSLA'])
st.header(f'News of {symbol}')
sn=StockNews(symbol,save_news=False)
df_news = sn.read_rss()
for i in range(10):
    st.subheader(f'News {i+1}')
    st.write(df_news['published'][i])
    st.write(df_news['title'][i])
    st.write(df_news['summary'][i])
    # title_sentiment = df_news['sentiment_title'][i]
    # st.write(f'Title Sentiment: {title_sentiment}')
    # news_sentiment = df_news['sentiment_summary']
    # st.write(f'News Sentiment {news_sentiment}')