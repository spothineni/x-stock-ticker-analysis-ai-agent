"""Stock analysis module that combines sentiment and price data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..utils.database import db_manager
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class StockAnalyzer:
    """Analyzes stock data combined with sentiment analysis."""
    
    def __init__(self):
        """Initialize stock analyzer."""
        self.scaler = StandardScaler()
        logger.info("Stock analyzer initialized")
    
    def calculate_technical_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for stock data."""
        df = stock_data.copy()
        
        # Simple Moving Averages
        df['sma_5'] = df['close_price'].rolling(window=5).mean()
        df['sma_10'] = df['close_price'].rolling(window=10).mean()
        df['sma_20'] = df['close_price'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close_price'].ewm(span=5).mean()
        df['ema_10'] = df['close_price'].ewm(span=10).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close_price'].rolling(window=20).mean()
        bb_std = df['close_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # RSI (Relative Strength Index)
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close_price'].ewm(span=12).mean()
        ema_26 = df['close_price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price change and volatility
        df['price_change'] = df['close_price'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def get_stock_data_with_indicators(self, ticker: str, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Get stock data with technical indicators."""
        try:
            stock_data = db_manager.get_stock_data_by_ticker(ticker, days_back)
            
            if not stock_data:
                logger.warning(f"No stock data found for {ticker}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': s.date,
                'open_price': s.open_price,
                'high_price': s.high_price,
                'low_price': s.low_price,
                'close_price': s.close_price,
                'volume': s.volume,
                'adj_close': s.adj_close
            } for s in stock_data])
            
            df = df.sort_values('date')
            df = df.reset_index(drop=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock data with indicators for {ticker}: {e}")
            return None
    
    def get_sentiment_data(self, ticker: str, days_back: int = 7) -> Optional[pd.DataFrame]:
        """Get sentiment data for a ticker."""
        try:
            session = db_manager.get_session()
            
            # Get sentiment analysis data
            from ..utils.database import SentimentAnalysis
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            sentiment_data = session.query(SentimentAnalysis).filter(
                SentimentAnalysis.ticker == ticker,
                SentimentAnalysis.analyzed_at >= cutoff_date
            ).all()
            
            session.close()
            
            if not sentiment_data:
                logger.warning(f"No sentiment data found for {ticker}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': s.analyzed_at.date(),
                'sentiment_score': s.sentiment_score,
                'confidence': s.confidence,
                'model_used': s.model_used
            } for s in sentiment_data])
            
            # Group by date and calculate daily averages
            daily_sentiment = df.groupby('date').agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'confidence': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'tweet_count', 'avg_confidence']
            daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
            
            return daily_sentiment
            
        except Exception as e:
            logger.error(f"Error getting sentiment data for {ticker}: {e}")
            return None
    
    def combine_stock_and_sentiment_data(self, ticker: str, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Combine stock data with sentiment data."""
        try:
            # Get stock data
            stock_df = self.get_stock_data_with_indicators(ticker, days_back)
            if stock_df is None:
                return None
            
            # Get sentiment data
            sentiment_df = self.get_sentiment_data(ticker, days_back)
            
            # Convert date columns to datetime for merging
            stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
            
            if sentiment_df is not None:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
                
                # Merge on date
                combined_df = pd.merge(stock_df, sentiment_df, on='date', how='left')
            else:
                # If no sentiment data, add empty columns
                combined_df = stock_df.copy()
                combined_df['avg_sentiment'] = 0.0
                combined_df['sentiment_std'] = 0.0
                combined_df['tweet_count'] = 0
                combined_df['avg_confidence'] = 0.0
            
            # Fill missing sentiment values
            combined_df['avg_sentiment'] = combined_df['avg_sentiment'].fillna(0.0)
            combined_df['sentiment_std'] = combined_df['sentiment_std'].fillna(0.0)
            combined_df['tweet_count'] = combined_df['tweet_count'].fillna(0)
            combined_df['avg_confidence'] = combined_df['avg_confidence'].fillna(0.0)
            
            # Sort by date
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining data for {ticker}: {e}")
            return None
    
    def calculate_correlation_analysis(self, ticker: str) -> Dict[str, Any]:
        """Calculate correlation between sentiment and stock price movements."""
        try:
            combined_df = self.combine_stock_and_sentiment_data(ticker)
            
            if combined_df is None or len(combined_df) < 5:
                return {
                    'ticker': ticker,
                    'correlation_sentiment_price': 0.0,
                    'correlation_sentiment_volume': 0.0,
                    'correlation_tweets_volatility': 0.0,
                    'p_value_sentiment_price': 1.0,
                    'sample_size': 0,
                    'analysis_date': datetime.utcnow()
                }
            
            # Calculate correlations
            sentiment_price_corr, sentiment_price_p = stats.pearsonr(
                combined_df['avg_sentiment'], combined_df['price_change'].fillna(0)
            )
            
            sentiment_volume_corr, sentiment_volume_p = stats.pearsonr(
                combined_df['avg_sentiment'], combined_df['volume_ratio'].fillna(1)
            )
            
            tweets_volatility_corr, tweets_volatility_p = stats.pearsonr(
                combined_df['tweet_count'], combined_df['volatility'].fillna(0)
            )
            
            return {
                'ticker': ticker,
                'correlation_sentiment_price': sentiment_price_corr,
                'correlation_sentiment_volume': sentiment_volume_corr,
                'correlation_tweets_volatility': tweets_volatility_corr,
                'p_value_sentiment_price': sentiment_price_p,
                'p_value_sentiment_volume': sentiment_volume_p,
                'p_value_tweets_volatility': tweets_volatility_p,
                'sample_size': len(combined_df),
                'analysis_date': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlations for {ticker}: {e}")
            return {
                'ticker': ticker,
                'correlation_sentiment_price': 0.0,
                'correlation_sentiment_volume': 0.0,
                'correlation_tweets_volatility': 0.0,
                'p_value_sentiment_price': 1.0,
                'sample_size': 0,
                'analysis_date': datetime.utcnow()
            }
    
    def calculate_support_resistance_levels(self, ticker: str) -> Dict[str, Any]:
        """Calculate support and resistance levels."""
        try:
            stock_df = self.get_stock_data_with_indicators(ticker, 30)
            
            if stock_df is None or len(stock_df) < 10:
                return {
                    'ticker': ticker,
                    'support_levels': [],
                    'resistance_levels': [],
                    'current_price': None,
                    'analysis_date': datetime.utcnow()
                }
            
            # Get recent price data
            prices = stock_df['close_price'].values
            highs = stock_df['high_price'].values
            lows = stock_df['low_price'].values
            
            # Find local minima (support) and maxima (resistance)
            from scipy.signal import argrelextrema
            
            # Find local minima (support levels)
            support_indices = argrelextrema(lows, np.less, order=3)[0]
            support_levels = [float(lows[i]) for i in support_indices]
            
            # Find local maxima (resistance levels)
            resistance_indices = argrelextrema(highs, np.greater, order=3)[0]
            resistance_levels = [float(highs[i]) for i in resistance_indices]
            
            # Sort levels
            support_levels.sort(reverse=True)  # Highest support first
            resistance_levels.sort()  # Lowest resistance first
            
            # Keep only the most relevant levels (top 3 of each)
            support_levels = support_levels[:3]
            resistance_levels = resistance_levels[:3]
            
            current_price = float(prices[-1])
            
            return {
                'ticker': ticker,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'current_price': current_price,
                'analysis_date': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance for {ticker}: {e}")
            return {
                'ticker': ticker,
                'support_levels': [],
                'resistance_levels': [],
                'current_price': None,
                'analysis_date': datetime.utcnow()
            }
    
    def calculate_volatility_analysis(self, ticker: str) -> Dict[str, Any]:
        """Calculate volatility metrics."""
        try:
            stock_df = self.get_stock_data_with_indicators(ticker, 30)
            
            if stock_df is None or len(stock_df) < 5:
                return {
                    'ticker': ticker,
                    'current_volatility': 0.0,
                    'avg_volatility': 0.0,
                    'volatility_percentile': 0.0,
                    'is_high_volatility': False,
                    'analysis_date': datetime.utcnow()
                }
            
            volatility = stock_df['volatility'].dropna()
            
            if len(volatility) == 0:
                return {
                    'ticker': ticker,
                    'current_volatility': 0.0,
                    'avg_volatility': 0.0,
                    'volatility_percentile': 0.0,
                    'is_high_volatility': False,
                    'analysis_date': datetime.utcnow()
                }
            
            current_volatility = float(volatility.iloc[-1])
            avg_volatility = float(volatility.mean())
            volatility_percentile = float(stats.percentileofscore(volatility, current_volatility))
            
            # Consider high volatility if above 75th percentile
            is_high_volatility = volatility_percentile > 75
            
            return {
                'ticker': ticker,
                'current_volatility': current_volatility,
                'avg_volatility': avg_volatility,
                'volatility_percentile': volatility_percentile,
                'is_high_volatility': is_high_volatility,
                'analysis_date': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {ticker}: {e}")
            return {
                'ticker': ticker,
                'current_volatility': 0.0,
                'avg_volatility': 0.0,
                'volatility_percentile': 0.0,
                'is_high_volatility': False,
                'analysis_date': datetime.utcnow()
            }
    
    def generate_daily_analysis(self, ticker: str) -> Dict[str, Any]:
        """Generate comprehensive daily analysis for a ticker."""
        try:
            logger.info(f"Generating daily analysis for {ticker}")
            
            # Get combined data
            combined_df = self.combine_stock_and_sentiment_data(ticker)
            
            if combined_df is None or len(combined_df) == 0:
                logger.warning(f"No data available for analysis of {ticker}")
                return None
            
            # Get latest data point
            latest_data = combined_df.iloc[-1]
            
            # Calculate various analyses
            correlation_analysis = self.calculate_correlation_analysis(ticker)
            support_resistance = self.calculate_support_resistance_levels(ticker)
            volatility_analysis = self.calculate_volatility_analysis(ticker)
            
            # Calculate price change
            if len(combined_df) >= 2:
                prev_close = combined_df.iloc[-2]['close_price']
                current_close = latest_data['close_price']
                price_change_percent = ((current_close - prev_close) / prev_close) * 100
            else:
                price_change_percent = 0.0
            
            # Calculate volume change
            if len(combined_df) >= 2:
                prev_volume = combined_df.iloc[-2]['volume']
                current_volume = latest_data['volume']
                volume_change_percent = ((current_volume - prev_volume) / prev_volume) * 100 if prev_volume > 0 else 0.0
            else:
                volume_change_percent = 0.0
            
            analysis_result = {
                'ticker': ticker,
                'analysis_date': datetime.utcnow(),
                'tweet_count': int(latest_data['tweet_count']),
                'avg_sentiment': float(latest_data['avg_sentiment']),
                'sentiment_std': float(latest_data['sentiment_std']),
                'price_change_percent': float(price_change_percent),
                'volume_change_percent': float(volume_change_percent),
                'current_price': float(latest_data['close_price']),
                'current_volume': int(latest_data['volume']),
                'correlation_analysis': correlation_analysis,
                'support_resistance': support_resistance,
                'volatility_analysis': volatility_analysis,
                'technical_indicators': {
                    'rsi': float(latest_data.get('rsi', 50)),
                    'macd': float(latest_data.get('macd', 0)),
                    'bb_position': self._calculate_bb_position(latest_data),
                    'volume_ratio': float(latest_data.get('volume_ratio', 1))
                }
            }
            
            logger.info(f"Daily analysis completed for {ticker}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error generating daily analysis for {ticker}: {e}")
            return None
    
    def _calculate_bb_position(self, data_row) -> float:
        """Calculate position within Bollinger Bands."""
        try:
            price = data_row['close_price']
            bb_upper = data_row.get('bb_upper')
            bb_lower = data_row.get('bb_lower')
            
            if bb_upper is None or bb_lower is None or bb_upper == bb_lower:
                return 0.5  # Middle position if bands not available
            
            # Calculate position (0 = at lower band, 1 = at upper band, 0.5 = middle)
            position = (price - bb_lower) / (bb_upper - bb_lower)
            return max(0, min(1, position))  # Clamp between 0 and 1
            
        except Exception:
            return 0.5