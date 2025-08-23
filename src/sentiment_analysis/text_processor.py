"""Text processing utilities for tweet analysis."""

import re
import string
from typing import List, Dict, Any
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class TextProcessor:
    """Processes and cleans text for sentiment analysis."""
    
    def __init__(self):
        """Initialize text processor."""
        # Common financial terms and their normalized forms
        self.financial_terms = {
            'bullish': 'positive',
            'bearish': 'negative',
            'moon': 'very_positive',
            'rocket': 'very_positive',
            'crash': 'very_negative',
            'dump': 'very_negative',
            'hodl': 'hold',
            'diamond hands': 'hold',
            'paper hands': 'sell',
            'to the moon': 'very_positive',
            'buy the dip': 'positive',
            'stonks': 'stocks'
        }
        
        # Stock-related keywords that indicate financial context
        self.stock_keywords = {
            'buy', 'sell', 'hold', 'long', 'short', 'calls', 'puts', 'options',
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'split',
            'ipo', 'merger', 'acquisition', 'breakout', 'support', 'resistance',
            'volume', 'volatility', 'bull', 'bear', 'market', 'trading',
            'investment', 'portfolio', 'shares', 'stock', 'equity'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions but keep the context
        text = re.sub(r'@\w+', '', text)
        
        # Keep hashtags but remove the # symbol
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize financial slang
        for term, normalized in self.financial_terms.items():
            text = text.replace(term, normalized)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_financial_context(self, text: str) -> Dict[str, Any]:
        """Extract financial context from text."""
        text_lower = text.lower()
        
        context = {
            'has_financial_keywords': False,
            'financial_keywords': [],
            'action_words': [],
            'sentiment_modifiers': [],
            'ticker_mentions': [],
            'financial_score': 0.0
        }
        
        # Find financial keywords
        found_keywords = []
        for keyword in self.stock_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        context['financial_keywords'] = found_keywords
        context['has_financial_keywords'] = len(found_keywords) > 0
        
        # Extract action words (buy, sell, hold, etc.)
        action_words = ['buy', 'sell', 'hold', 'long', 'short']
        context['action_words'] = [word for word in action_words if word in text_lower]
        
        # Extract sentiment modifiers
        positive_modifiers = ['very', 'extremely', 'super', 'amazing', 'great', 'excellent']
        negative_modifiers = ['very', 'extremely', 'terrible', 'awful', 'horrible', 'bad']
        
        for modifier in positive_modifiers:
            if modifier in text_lower:
                context['sentiment_modifiers'].append(('positive', modifier))
        
        for modifier in negative_modifiers:
            if modifier in text_lower:
                context['sentiment_modifiers'].append(('negative', modifier))
        
        # Extract ticker mentions
        ticker_pattern = r'\$([A-Z]{1,5})'
        tickers = re.findall(ticker_pattern, text.upper())
        context['ticker_mentions'] = list(set(tickers))
        
        # Calculate financial relevance score
        financial_score = 0.0
        financial_score += len(found_keywords) * 0.1
        financial_score += len(context['action_words']) * 0.2
        financial_score += len(context['ticker_mentions']) * 0.3
        financial_score += len(context['sentiment_modifiers']) * 0.1
        
        context['financial_score'] = min(financial_score, 1.0)
        
        return context
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """Preprocess text specifically for sentiment analysis."""
        # Clean the text
        cleaned = self.clean_text(text)
        
        # Remove punctuation except for emoticons and important financial symbols
        # Keep $ for tickers, % for percentages, and basic emoticons
        important_chars = ['$', '%', ':', ')', '(', '!', '?']
        
        # Create a translation table that removes punctuation except important chars
        translator = str.maketrans('', '', ''.join(c for c in string.punctuation if c not in important_chars))
        cleaned = cleaned.translate(translator)
        
        # Normalize repeated characters (e.g., "sooooo" -> "so")
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
        
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def extract_price_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Extract price mentions and targets from text."""
        price_mentions = []
        
        # Pattern for price mentions: $123, $123.45, 123$, etc.
        price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # $123.45
            r'(\d+(?:\.\d{2})?)\s*(?:dollars?|\$)',  # 123 dollars, 123$
            r'price\s+(?:target|of|at)\s+\$?(\d+(?:\.\d{2})?)',  # price target $123
            r'target\s+\$?(\d+(?:\.\d{2})?)',  # target $123
        ]
        
        for pattern in price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                price = float(match.group(1))
                price_mentions.append({
                    'price': price,
                    'context': match.group(0),
                    'position': match.span()
                })
        
        return price_mentions
    
    def extract_percentage_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Extract percentage mentions from text."""
        percentage_mentions = []
        
        # Pattern for percentage mentions: 10%, +5%, -3.5%
        percentage_pattern = r'([+-]?\d+(?:\.\d+)?)\s*%'
        
        matches = re.finditer(percentage_pattern, text)
        for match in matches:
            percentage = float(match.group(1))
            percentage_mentions.append({
                'percentage': percentage,
                'context': match.group(0),
                'position': match.span()
            })
        
        return percentage_mentions
    
    def is_financial_relevant(self, text: str, threshold: float = 0.3) -> bool:
        """Determine if text is financially relevant."""
        context = self.extract_financial_context(text)
        return context['financial_score'] >= threshold
    
    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract features that can help with sentiment analysis."""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'financial_context': self.extract_financial_context(text),
            'price_mentions': self.extract_price_mentions(text),
            'percentage_mentions': self.extract_percentage_mentions(text),
        }
        
        return features