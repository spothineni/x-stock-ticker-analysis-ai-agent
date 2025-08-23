"""Database utilities and models."""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)
Base = declarative_base()

class Tweet(Base):
    """Tweet data model."""
    __tablename__ = 'tweets'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String, unique=True, nullable=False)
    text = Column(Text, nullable=False)
    author_id = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    retweet_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    quote_count = Column(Integer, default=0)
    collected_at = Column(DateTime, default=datetime.utcnow)

class SentimentAnalysis(Base):
    """Sentiment analysis results model."""
    __tablename__ = 'sentiment_analysis'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    sentiment_label = Column(String, nullable=False)  # positive, negative, neutral
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

class StockData(Base):
    """Stock price data model."""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adj_close = Column(Float, nullable=False)

class DailyAnalysis(Base):
    """Daily analysis results model."""
    __tablename__ = 'daily_analysis'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    analysis_date = Column(DateTime, nullable=False)
    tweet_count = Column(Integer, nullable=False)
    avg_sentiment = Column(Float, nullable=False)
    sentiment_std = Column(Float, nullable=False)
    price_change_percent = Column(Float, nullable=False)
    volume_change_percent = Column(Float, nullable=False)
    predicted_range_low = Column(Float)
    predicted_range_high = Column(Float)
    prediction_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database manager for the stock sentiment analyzer."""
    
    def __init__(self):
        """Initialize database connection."""
        db_config = config.database_config
        self.engine = create_engine(db_config['url'], echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database initialized successfully")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def save_tweets(self, tweets: List[Dict[str, Any]]) -> int:
        """Save tweets to database."""
        session = self.get_session()
        saved_count = 0
        
        try:
            for tweet_data in tweets:
                # Check if tweet already exists
                existing = session.query(Tweet).filter_by(tweet_id=tweet_data['tweet_id']).first()
                if existing:
                    continue
                
                tweet = Tweet(**tweet_data)
                session.add(tweet)
                saved_count += 1
            
            session.commit()
            logger.info(f"Saved {saved_count} new tweets to database")
            return saved_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving tweets: {e}")
            raise
        finally:
            session.close()
    
    def save_sentiment_analysis(self, sentiment_data: List[Dict[str, Any]]) -> int:
        """Save sentiment analysis results to database."""
        session = self.get_session()
        saved_count = 0
        
        try:
            for data in sentiment_data:
                sentiment = SentimentAnalysis(**data)
                session.add(sentiment)
                saved_count += 1
            
            session.commit()
            logger.info(f"Saved {saved_count} sentiment analysis results")
            return saved_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving sentiment analysis: {e}")
            raise
        finally:
            session.close()
    
    def save_stock_data(self, stock_data: List[Dict[str, Any]]) -> int:
        """Save stock data to database."""
        session = self.get_session()
        saved_count = 0
        
        try:
            for data in stock_data:
                # Check if data already exists
                existing = session.query(StockData).filter_by(
                    ticker=data['ticker'], 
                    date=data['date']
                ).first()
                if existing:
                    continue
                
                stock = StockData(**data)
                session.add(stock)
                saved_count += 1
            
            session.commit()
            logger.info(f"Saved {saved_count} new stock data records")
            return saved_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving stock data: {e}")
            raise
        finally:
            session.close()
    
    def save_daily_analysis(self, analysis_data: Dict[str, Any]) -> int:
        """Save daily analysis results to database."""
        session = self.get_session()
        
        try:
            analysis = DailyAnalysis(**analysis_data)
            session.add(analysis)
            session.commit()
            logger.info(f"Saved daily analysis for {analysis_data['ticker']}")
            return analysis.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving daily analysis: {e}")
            raise
        finally:
            session.close()
    
    def get_tweets_by_ticker(self, ticker: str, days_back: int = 7) -> List[Tweet]:
        """Get tweets for a specific ticker."""
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            tweets = session.query(Tweet).filter(
                Tweet.ticker == ticker,
                Tweet.created_at >= cutoff_date
            ).all()
            
            return tweets
        finally:
            session.close()
    
    def get_stock_data_by_ticker(self, ticker: str, days_back: int = 30) -> List[StockData]:
        """Get stock data for a specific ticker."""
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            stock_data = session.query(StockData).filter(
                StockData.ticker == ticker,
                StockData.date >= cutoff_date
            ).order_by(StockData.date.desc()).all()
            
            return stock_data
        finally:
            session.close()

# Global database manager instance
db_manager = DatabaseManager()