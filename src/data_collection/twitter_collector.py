"""Twitter/X data collection module."""

import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import tweepy
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class TwitterCollector:
    """Collects tweets related to stock tickers."""
    
    def __init__(self):
        """Initialize Twitter API client."""
        twitter_config = config.twitter_config
        
        # Check if we have the required credentials
        if not twitter_config.get('bearer_token'):
            logger.warning("Twitter Bearer Token not found. Please set TWITTER_BEARER_TOKEN in .env file")
            self.client = None
            return
        
        try:
            # Initialize Twitter API v2 client
            self.client = tweepy.Client(
                bearer_token=twitter_config['bearer_token'],
                consumer_key=twitter_config.get('api_key'),
                consumer_secret=twitter_config.get('api_secret'),
                access_token=twitter_config.get('access_token'),
                access_token_secret=twitter_config.get('access_token_secret'),
                wait_on_rate_limit=True
            )
            
            # Test the connection
            me = self.client.get_me()
            if me:
                logger.info("Twitter API connection established successfully")
            else:
                logger.info("Twitter API connection established (using app-only auth)")
                
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            self.client = None
    
    def extract_stock_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from tweet text."""
        # Pattern to match $TICKER format
        ticker_pattern = r'\$([A-Z]{1,5})'
        tickers = re.findall(ticker_pattern, text.upper())
        
        # Filter to only include configured tickers
        configured_tickers = config.stock_tickers
        valid_tickers = [ticker for ticker in tickers if ticker in configured_tickers]
        
        return list(set(valid_tickers))  # Remove duplicates
    
    def clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags for cleaner sentiment analysis
        # but keep the ticker symbols
        text = re.sub(r'@\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def search_tweets_for_ticker(self, ticker: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for tweets mentioning a specific stock ticker."""
        if not self.client:
            logger.error("Twitter API client not initialized")
            return []
        
        try:
            # Search query for the ticker
            query = f"${ticker} -is:retweet lang:en"
            
            # Get tweets from the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=config.get('data_collection.twitter.search_days_back', 1))
            
            logger.info(f"Searching for tweets with ticker ${ticker}")
            
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                max_results=min(max_results, 100),  # API limit
                start_time=start_time,
                end_time=end_time
            ).flatten(limit=max_results)
            
            tweet_data = []
            for tweet in tweets:
                # Extract all tickers from the tweet (might mention multiple)
                mentioned_tickers = self.extract_stock_tickers(tweet.text)
                
                # Only process if our target ticker is mentioned
                if ticker in mentioned_tickers:
                    tweet_info = {
                        'tweet_id': str(tweet.id),
                        'text': self.clean_tweet_text(tweet.text),
                        'author_id': str(tweet.author_id),
                        'created_at': tweet.created_at,
                        'ticker': ticker,
                        'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                        'like_count': tweet.public_metrics.get('like_count', 0),
                        'reply_count': tweet.public_metrics.get('reply_count', 0),
                        'quote_count': tweet.public_metrics.get('quote_count', 0),
                    }
                    tweet_data.append(tweet_info)
            
            logger.info(f"Found {len(tweet_data)} tweets for ${ticker}")
            return tweet_data
            
        except tweepy.TooManyRequests:
            logger.warning(f"Rate limit reached for ticker ${ticker}. Waiting...")
            time.sleep(15 * 60)  # Wait 15 minutes
            return []
        except Exception as e:
            logger.error(f"Error searching tweets for ${ticker}: {e}")
            return []
    
    def collect_tweets_for_all_tickers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect tweets for all configured stock tickers."""
        if not self.client:
            logger.error("Twitter API client not initialized")
            return {}
        
        all_tweets = {}
        max_tweets_per_ticker = config.get('data_collection.twitter.max_tweets_per_ticker', 100)
        rate_limit_delay = config.get('data_collection.twitter.rate_limit_delay', 15)
        
        for ticker in config.stock_tickers:
            logger.info(f"Collecting tweets for ${ticker}")
            
            tweets = self.search_tweets_for_ticker(ticker, max_tweets_per_ticker)
            all_tweets[ticker] = tweets
            
            # Add delay to respect rate limits
            if ticker != config.stock_tickers[-1]:  # Don't delay after the last ticker
                logger.info(f"Waiting {rate_limit_delay} seconds before next ticker...")
                time.sleep(rate_limit_delay)
        
        total_tweets = sum(len(tweets) for tweets in all_tweets.values())
        logger.info(f"Collected {total_tweets} total tweets across {len(config.stock_tickers)} tickers")
        
        return all_tweets
    
    def get_trending_stocks(self) -> List[str]:
        """Get trending stock-related topics (if available)."""
        if not self.client:
            return []
        
        try:
            # This is a placeholder - Twitter's trending topics API has limitations
            # In a real implementation, you might use different approaches:
            # 1. Monitor specific financial Twitter accounts
            # 2. Use Twitter's Academic Research API for more comprehensive data
            # 3. Combine with other data sources
            
            logger.info("Trending stocks feature not implemented yet")
            return []
            
        except Exception as e:
            logger.error(f"Error getting trending stocks: {e}")
            return []