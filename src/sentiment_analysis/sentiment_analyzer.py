"""Sentiment analysis module for financial tweets."""

from typing import List, Dict, Any, Tuple
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .text_processor import TextProcessor
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment of financial tweets using multiple models."""
    
    def __init__(self):
        """Initialize sentiment analyzer with multiple models."""
        self.text_processor = TextProcessor()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Add financial terms to VADER lexicon
        self._enhance_vader_lexicon()
        
        # Model weights for ensemble
        self.model_weights = {
            'textblob': 0.3,
            'vader': 0.4,
            'financial_rules': 0.3
        }
        
        logger.info("Sentiment analyzer initialized with multiple models")
    
    def _enhance_vader_lexicon(self):
        """Add financial-specific terms to VADER lexicon."""
        financial_lexicon = {
            'bullish': 2.5,
            'bearish': -2.5,
            'moon': 3.0,
            'rocket': 2.8,
            'crash': -3.0,
            'dump': -2.5,
            'hodl': 1.5,
            'diamond hands': 2.0,
            'paper hands': -1.5,
            'buy the dip': 2.0,
            'stonks': 1.0,
            'gains': 2.5,
            'losses': -2.5,
            'profit': 2.0,
            'loss': -2.0,
            'breakout': 2.5,
            'support': 1.5,
            'resistance': -1.0,
            'oversold': 1.5,
            'overbought': -1.5,
            'squeeze': 2.0,
            'gamma': 1.5,
            'short squeeze': 3.0,
            'rug pull': -3.5,
            'pump': 2.0,
            'dump': -2.5,
            'fomo': 1.0,
            'yolo': 2.0,
            'diamond': 2.0,
            'paper': -1.0,
            'tendies': 2.5,
            'baghold': -2.0,
            'bagholder': -2.0
        }
        
        # Update VADER lexicon
        self.vader_analyzer.lexicon.update(financial_lexicon)
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'confidence': abs(polarity)
            }
        except Exception as e:
            logger.error(f"TextBlob analysis error: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5, 'confidence': 0.0}
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Convert compound score to -1 to 1 range (it's already in this range)
            compound = scores['compound']
            
            return {
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'confidence': abs(compound)
            }
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'confidence': 0.0}
    
    def analyze_with_financial_rules(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using financial-specific rules."""
        try:
            text_lower = text.lower()
            sentiment_score = 0.0
            confidence = 0.0
            
            # Financial action words
            bullish_actions = ['buy', 'long', 'call', 'bullish', 'up', 'rise', 'gain']
            bearish_actions = ['sell', 'short', 'put', 'bearish', 'down', 'fall', 'loss']
            
            # Count bullish vs bearish signals
            bullish_count = sum(1 for word in bullish_actions if word in text_lower)
            bearish_count = sum(1 for word in bearish_actions if word in text_lower)
            
            # Price direction indicators
            if any(phrase in text_lower for phrase in ['to the moon', 'rocket', 'moon']):
                sentiment_score += 0.8
                confidence += 0.3
            
            if any(phrase in text_lower for phrase in ['crash', 'dump', 'rug pull']):
                sentiment_score -= 0.8
                confidence += 0.3
            
            # Percentage mentions
            percentages = self.text_processor.extract_percentage_mentions(text)
            for pct in percentages:
                if pct['percentage'] > 0:
                    sentiment_score += min(pct['percentage'] / 100, 0.5)
                    confidence += 0.1
                else:
                    sentiment_score += max(pct['percentage'] / 100, -0.5)
                    confidence += 0.1
            
            # Action word balance
            if bullish_count > bearish_count:
                sentiment_score += (bullish_count - bearish_count) * 0.2
                confidence += 0.2
            elif bearish_count > bullish_count:
                sentiment_score -= (bearish_count - bullish_count) * 0.2
                confidence += 0.2
            
            # Normalize sentiment score to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            confidence = min(1.0, confidence)
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count
            }
            
        except Exception as e:
            logger.error(f"Financial rules analysis error: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'bullish_signals': 0, 'bearish_signals': 0}
    
    def get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis using ensemble of models."""
        if not text or not text.strip():
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'model_scores': {},
                'financial_relevance': 0.0
            }
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_for_sentiment(text)
        
        # Check financial relevance
        financial_relevance = self.text_processor.extract_financial_context(text)['financial_score']
        
        # Get scores from all models
        textblob_result = self.analyze_with_textblob(processed_text)
        vader_result = self.analyze_with_vader(processed_text)
        financial_result = self.analyze_with_financial_rules(text)
        
        # Calculate ensemble score
        ensemble_score = (
            textblob_result['polarity'] * self.model_weights['textblob'] +
            vader_result['compound'] * self.model_weights['vader'] +
            financial_result['sentiment_score'] * self.model_weights['financial_rules']
        )
        
        # Calculate ensemble confidence
        ensemble_confidence = (
            textblob_result['confidence'] * self.model_weights['textblob'] +
            vader_result['confidence'] * self.model_weights['vader'] +
            financial_result['confidence'] * self.model_weights['financial_rules']
        )
        
        # Adjust confidence based on financial relevance
        if financial_relevance > 0.5:
            ensemble_confidence *= 1.2  # Boost confidence for financially relevant tweets
        
        ensemble_confidence = min(1.0, ensemble_confidence)
        
        return {
            'sentiment_score': ensemble_score,
            'sentiment_label': self.get_sentiment_label(ensemble_score),
            'confidence': ensemble_confidence,
            'financial_relevance': financial_relevance,
            'model_scores': {
                'textblob': textblob_result,
                'vader': vader_result,
                'financial_rules': financial_result
            },
            'text_features': self.text_processor.extract_sentiment_features(text)
        }
    
    def analyze_tweet_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of tweets."""
        results = []
        
        for tweet in tweets:
            try:
                sentiment_result = self.analyze_sentiment(tweet['text'])
                
                # Create sentiment analysis record
                sentiment_record = {
                    'tweet_id': tweet['tweet_id'],
                    'ticker': tweet['ticker'],
                    'sentiment_score': sentiment_result['sentiment_score'],
                    'sentiment_label': sentiment_result['sentiment_label'],
                    'confidence': sentiment_result['confidence'],
                    'model_used': 'ensemble',
                    'financial_relevance': sentiment_result['financial_relevance'],
                    'model_details': sentiment_result['model_scores']
                }
                
                results.append(sentiment_record)
                
            except Exception as e:
                logger.error(f"Error analyzing tweet {tweet.get('tweet_id', 'unknown')}: {e}")
                # Add neutral sentiment for failed analysis
                results.append({
                    'tweet_id': tweet['tweet_id'],
                    'ticker': tweet['ticker'],
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'model_used': 'ensemble',
                    'financial_relevance': 0.0,
                    'model_details': {}
                })
        
        logger.info(f"Analyzed sentiment for {len(results)} tweets")
        return results
    
    def get_ticker_sentiment_summary(self, sentiment_results: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
        """Get sentiment summary for a specific ticker."""
        ticker_sentiments = [r for r in sentiment_results if r['ticker'] == ticker]
        
        if not ticker_sentiments:
            return {
                'ticker': ticker,
                'tweet_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'avg_confidence': 0.0,
                'financial_relevance': 0.0
            }
        
        scores = [s['sentiment_score'] for s in ticker_sentiments]
        confidences = [s['confidence'] for s in ticker_sentiments]
        relevances = [s.get('financial_relevance', 0.0) for s in ticker_sentiments]
        
        # Count sentiment labels
        positive_count = sum(1 for s in ticker_sentiments if s['sentiment_label'] == 'positive')
        negative_count = sum(1 for s in ticker_sentiments if s['sentiment_label'] == 'negative')
        neutral_count = sum(1 for s in ticker_sentiments if s['sentiment_label'] == 'neutral')
        
        total_count = len(ticker_sentiments)
        
        return {
            'ticker': ticker,
            'tweet_count': total_count,
            'avg_sentiment': np.mean(scores),
            'sentiment_std': np.std(scores),
            'positive_ratio': positive_count / total_count,
            'negative_ratio': negative_count / total_count,
            'neutral_ratio': neutral_count / total_count,
            'avg_confidence': np.mean(confidences),
            'avg_financial_relevance': np.mean(relevances),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }