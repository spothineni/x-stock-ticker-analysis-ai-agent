"""Report generation module for daily stock analysis and forecasts."""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.database import db_manager
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ReportGenerator:
    """Generates comprehensive daily reports with analysis and forecasts."""
    
    def __init__(self):
        """Initialize report generator."""
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Report generator initialized")
    
    def generate_daily_report(self, report_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive daily report."""
        if report_date is None:
            report_date = datetime.utcnow()
        
        logger.info(f"Generating daily report for {report_date.date()}")
        
        report = {
            'report_date': report_date,
            'generated_at': datetime.utcnow(),
            'market_summary': self._generate_market_summary(report_date),
            'ticker_analyses': self._generate_ticker_analyses(report_date),
            'sentiment_overview': self._generate_sentiment_overview(report_date),
            'predictions_summary': self._generate_predictions_summary(report_date),
            'key_insights': self._generate_key_insights(report_date),
            'recommendations': self._generate_recommendations(report_date)
        }
        
        # Save report
        self._save_daily_report(report)
        
        # Generate visualizations
        self._generate_report_visualizations(report)
        
        logger.info("Daily report generation completed")
        return report
    
    def _generate_market_summary(self, report_date: datetime) -> Dict[str, Any]:
        """Generate overall market summary."""
        try:
            session = db_manager.get_session()
            from ..utils.database import DailyAnalysis, StockData
            
            # Get today's analysis data
            today = report_date.date()
            analyses = session.query(DailyAnalysis).filter(
                DailyAnalysis.analysis_date >= today
            ).all()
            
            if not analyses:
                session.close()
                return {
                    'total_tickers': len(config.stock_tickers),
                    'analyzed_tickers': 0,
                    'avg_sentiment': 0.0,
                    'positive_sentiment_ratio': 0.0,
                    'total_tweets': 0,
                    'market_trend': 'neutral'
                }
            
            # Calculate summary statistics
            total_tweets = sum(a.tweet_count for a in analyses)
            avg_sentiment = sum(a.avg_sentiment * a.tweet_count for a in analyses) / total_tweets if total_tweets > 0 else 0
            positive_count = sum(1 for a in analyses if a.avg_sentiment > 0.1)
            
            # Determine market trend
            avg_price_change = sum(a.price_change_percent for a in analyses) / len(analyses)
            if avg_price_change > 1:
                market_trend = 'bullish'
            elif avg_price_change < -1:
                market_trend = 'bearish'
            else:
                market_trend = 'neutral'
            
            session.close()
            
            return {
                'total_tickers': len(config.stock_tickers),
                'analyzed_tickers': len(analyses),
                'avg_sentiment': avg_sentiment,
                'positive_sentiment_ratio': positive_count / len(analyses) if analyses else 0,
                'total_tweets': total_tweets,
                'avg_price_change': avg_price_change,
                'market_trend': market_trend
            }
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return {
                'total_tickers': len(config.stock_tickers),
                'analyzed_tickers': 0,
                'avg_sentiment': 0.0,
                'positive_sentiment_ratio': 0.0,
                'total_tweets': 0,
                'market_trend': 'neutral'
            }
    
    def _generate_ticker_analyses(self, report_date: datetime) -> Dict[str, Dict[str, Any]]:
        """Generate detailed analysis for each ticker."""
        ticker_analyses = {}
        
        try:
            session = db_manager.get_session()
            from ..utils.database import DailyAnalysis
            
            today = report_date.date()
            
            for ticker in config.stock_tickers:
                # Get today's analysis
                analysis = session.query(DailyAnalysis).filter(
                    DailyAnalysis.ticker == ticker,
                    DailyAnalysis.analysis_date >= today
                ).first()
                
                if analysis:
                    ticker_analysis = {
                        'ticker': ticker,
                        'tweet_count': analysis.tweet_count,
                        'avg_sentiment': analysis.avg_sentiment,
                        'sentiment_std': analysis.sentiment_std,
                        'price_change_percent': analysis.price_change_percent,
                        'volume_change_percent': analysis.volume_change_percent,
                        'predicted_range_low': analysis.predicted_range_low,
                        'predicted_range_high': analysis.predicted_range_high,
                        'prediction_confidence': analysis.prediction_confidence,
                        'sentiment_label': self._get_sentiment_label(analysis.avg_sentiment),
                        'price_trend': self._get_price_trend(analysis.price_change_percent),
                        'volume_trend': self._get_volume_trend(analysis.volume_change_percent)
                    }
                else:
                    ticker_analysis = {
                        'ticker': ticker,
                        'tweet_count': 0,
                        'avg_sentiment': 0.0,
                        'sentiment_std': 0.0,
                        'price_change_percent': 0.0,
                        'volume_change_percent': 0.0,
                        'predicted_range_low': None,
                        'predicted_range_high': None,
                        'prediction_confidence': None,
                        'sentiment_label': 'neutral',
                        'price_trend': 'neutral',
                        'volume_trend': 'neutral'
                    }
                
                ticker_analyses[ticker] = ticker_analysis
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error generating ticker analyses: {e}")
        
        return ticker_analyses
    
    def _generate_sentiment_overview(self, report_date: datetime) -> Dict[str, Any]:
        """Generate sentiment analysis overview."""
        try:
            session = db_manager.get_session()
            from ..utils.database import SentimentAnalysis
            
            # Get sentiment data from the last 24 hours
            cutoff_time = report_date - timedelta(hours=24)
            
            sentiment_data = session.query(SentimentAnalysis).filter(
                SentimentAnalysis.analyzed_at >= cutoff_time
            ).all()
            
            session.close()
            
            if not sentiment_data:
                return {
                    'total_analyzed': 0,
                    'avg_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'most_positive_ticker': None,
                    'most_negative_ticker': None,
                    'highest_confidence': 0.0
                }
            
            # Calculate sentiment statistics
            sentiments = [s.sentiment_score for s in sentiment_data]
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Count sentiment labels
            positive_count = sum(1 for s in sentiment_data if s.sentiment_label == 'positive')
            negative_count = sum(1 for s in sentiment_data if s.sentiment_label == 'negative')
            neutral_count = sum(1 for s in sentiment_data if s.sentiment_label == 'neutral')
            
            # Find most positive and negative tickers
            ticker_sentiments = {}
            for s in sentiment_data:
                if s.ticker not in ticker_sentiments:
                    ticker_sentiments[s.ticker] = []
                ticker_sentiments[s.ticker].append(s.sentiment_score)
            
            ticker_avg_sentiments = {
                ticker: sum(scores) / len(scores)
                for ticker, scores in ticker_sentiments.items()
            }
            
            most_positive_ticker = max(ticker_avg_sentiments.items(), key=lambda x: x[1])[0] if ticker_avg_sentiments else None
            most_negative_ticker = min(ticker_avg_sentiments.items(), key=lambda x: x[1])[0] if ticker_avg_sentiments else None
            
            # Highest confidence
            highest_confidence = max(s.confidence for s in sentiment_data)
            
            return {
                'total_analyzed': len(sentiment_data),
                'avg_sentiment': avg_sentiment,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'ticker_sentiments': ticker_avg_sentiments,
                'most_positive_ticker': most_positive_ticker,
                'most_negative_ticker': most_negative_ticker,
                'highest_confidence': highest_confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment overview: {e}")
            return {
                'total_analyzed': 0,
                'avg_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'most_positive_ticker': None,
                'most_negative_ticker': None,
                'highest_confidence': 0.0
            }
    
    def _generate_predictions_summary(self, report_date: datetime) -> Dict[str, Any]:
        """Generate predictions summary."""
        try:
            session = db_manager.get_session()
            from ..utils.database import DailyAnalysis
            
            today = report_date.date()
            analyses = session.query(DailyAnalysis).filter(
                DailyAnalysis.analysis_date >= today,
                DailyAnalysis.predicted_range_low.isnot(None)
            ).all()
            
            session.close()
            
            if not analyses:
                return {
                    'predictions_available': 0,
                    'avg_confidence': 0.0,
                    'bullish_predictions': 0,
                    'bearish_predictions': 0,
                    'neutral_predictions': 0
                }
            
            # Calculate prediction statistics
            confidences = [a.prediction_confidence for a in analyses if a.prediction_confidence]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Count prediction directions (based on range midpoint vs current price)
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for analysis in analyses:
                if analysis.predicted_range_low and analysis.predicted_range_high:
                    range_midpoint = (analysis.predicted_range_low + analysis.predicted_range_high) / 2
                    # We'd need current price to compare, for now use price change
                    if analysis.price_change_percent > 1:
                        bullish_count += 1
                    elif analysis.price_change_percent < -1:
                        bearish_count += 1
                    else:
                        neutral_count += 1
            
            return {
                'predictions_available': len(analyses),
                'avg_confidence': avg_confidence,
                'bullish_predictions': bullish_count,
                'bearish_predictions': bearish_count,
                'neutral_predictions': neutral_count
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions summary: {e}")
            return {
                'predictions_available': 0,
                'avg_confidence': 0.0,
                'bullish_predictions': 0,
                'bearish_predictions': 0,
                'neutral_predictions': 0
            }
    
    def _generate_key_insights(self, report_date: datetime) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        try:
            session = db_manager.get_session()
            from ..utils.database import DailyAnalysis
            
            today = report_date.date()
            analyses = session.query(DailyAnalysis).filter(
                DailyAnalysis.analysis_date >= today
            ).all()
            
            session.close()
            
            if not analyses:
                return ["No analysis data available for today."]
            
            # High sentiment vs price movement correlation
            high_sentiment_tickers = [a for a in analyses if a.avg_sentiment > 0.3 and a.tweet_count > 5]
            if high_sentiment_tickers:
                ticker_names = [a.ticker for a in high_sentiment_tickers]
                insights.append(f"High positive sentiment detected for: {', '.join(ticker_names)}")
            
            # High volume activity
            high_volume_tickers = [a for a in analyses if a.volume_change_percent > 50]
            if high_volume_tickers:
                ticker_names = [a.ticker for a in high_volume_tickers]
                insights.append(f"Unusual volume activity in: {', '.join(ticker_names)}")
            
            # Strong price movements
            strong_movers = [a for a in analyses if abs(a.price_change_percent) > 3]
            if strong_movers:
                up_movers = [a.ticker for a in strong_movers if a.price_change_percent > 3]
                down_movers = [a.ticker for a in strong_movers if a.price_change_percent < -3]
                
                if up_movers:
                    insights.append(f"Strong upward movement in: {', '.join(up_movers)}")
                if down_movers:
                    insights.append(f"Strong downward movement in: {', '.join(down_movers)}")
            
            # Sentiment-price divergence
            divergent_tickers = []
            for a in analyses:
                if a.avg_sentiment > 0.2 and a.price_change_percent < -2:
                    divergent_tickers.append(f"{a.ticker} (positive sentiment, negative price)")
                elif a.avg_sentiment < -0.2 and a.price_change_percent > 2:
                    divergent_tickers.append(f"{a.ticker} (negative sentiment, positive price)")
            
            if divergent_tickers:
                insights.append(f"Sentiment-price divergence detected: {', '.join(divergent_tickers)}")
            
            # Low activity warning
            low_activity_tickers = [a.ticker for a in analyses if a.tweet_count < 3]
            if len(low_activity_tickers) > len(config.stock_tickers) / 2:
                insights.append("Low social media activity detected across multiple tickers - predictions may be less reliable")
            
            if not insights:
                insights.append("Market showing normal activity patterns with no significant anomalies detected.")
            
        except Exception as e:
            logger.error(f"Error generating key insights: {e}")
            insights.append("Error generating insights - please check system logs.")
        
        return insights
    
    def _generate_recommendations(self, report_date: datetime) -> List[Dict[str, str]]:
        """Generate trading recommendations based on analysis."""
        recommendations = []
        
        try:
            session = db_manager.get_session()
            from ..utils.database import DailyAnalysis
            
            today = report_date.date()
            analyses = session.query(DailyAnalysis).filter(
                DailyAnalysis.analysis_date >= today
            ).all()
            
            session.close()
            
            for analysis in analyses:
                if analysis.tweet_count < 3:
                    continue  # Skip tickers with low social activity
                
                recommendation = {
                    'ticker': analysis.ticker,
                    'action': 'hold',
                    'confidence': 'low',
                    'reason': 'Insufficient data'
                }
                
                # Determine recommendation based on sentiment and predictions
                if analysis.avg_sentiment > 0.3 and analysis.prediction_confidence and analysis.prediction_confidence > 0.6:
                    if analysis.predicted_range_high and analysis.predicted_range_low:
                        range_midpoint = (analysis.predicted_range_high + analysis.predicted_range_low) / 2
                        # Assuming current price is close to recent close, use price change as proxy
                        if analysis.price_change_percent > 0:
                            recommendation['action'] = 'buy'
                            recommendation['confidence'] = 'medium' if analysis.prediction_confidence > 0.7 else 'low'
                            recommendation['reason'] = f"Positive sentiment ({analysis.avg_sentiment:.2f}) with bullish prediction"
                
                elif analysis.avg_sentiment < -0.3 and analysis.prediction_confidence and analysis.prediction_confidence > 0.6:
                    recommendation['action'] = 'sell'
                    recommendation['confidence'] = 'medium' if analysis.prediction_confidence > 0.7 else 'low'
                    recommendation['reason'] = f"Negative sentiment ({analysis.avg_sentiment:.2f}) with bearish prediction"
                
                elif abs(analysis.avg_sentiment) < 0.1:
                    recommendation['action'] = 'hold'
                    recommendation['confidence'] = 'medium'
                    recommendation['reason'] = "Neutral sentiment - wait for clearer signals"
                
                recommendations.append(recommendation)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label."""
        if sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_price_trend(self, price_change: float) -> str:
        """Convert price change to trend label."""
        if price_change > 1:
            return 'up'
        elif price_change < -1:
            return 'down'
        else:
            return 'neutral'
    
    def _get_volume_trend(self, volume_change: float) -> str:
        """Convert volume change to trend label."""
        if volume_change > 20:
            return 'high'
        elif volume_change < -20:
            return 'low'
        else:
            return 'normal'
    
    def _save_daily_report(self, report: Dict[str, Any]):
        """Save daily report to file."""
        try:
            report_date = report['report_date']
            filename = f"daily_report_{report_date.strftime('%Y%m%d')}.json"
            filepath = self.reports_dir / filename
            
            # Convert datetime objects to strings for JSON serialization
            json_report = self._convert_datetime_to_string(report)
            
            with open(filepath, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            
            logger.info(f"Daily report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving daily report: {e}")
    
    def _generate_report_visualizations(self, report: Dict[str, Any]):
        """Generate visualizations for the report."""
        try:
            report_date = report['report_date']
            date_str = report_date.strftime('%Y%m%d')
            
            # Create visualizations directory
            viz_dir = self.reports_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Sentiment Distribution Chart
            self._create_sentiment_distribution_chart(report, viz_dir, date_str)
            
            # 2. Price Change vs Sentiment Scatter Plot
            self._create_sentiment_price_scatter(report, viz_dir, date_str)
            
            # 3. Ticker Performance Summary
            self._create_ticker_performance_chart(report, viz_dir, date_str)
            
            logger.info(f"Report visualizations generated for {date_str}")
            
        except Exception as e:
            logger.error(f"Error generating report visualizations: {e}")
    
    def _create_sentiment_distribution_chart(self, report: Dict[str, Any], viz_dir: Path, date_str: str):
        """Create sentiment distribution pie chart."""
        try:
            sentiment_overview = report['sentiment_overview']
            distribution = sentiment_overview['sentiment_distribution']
            
            if sum(distribution.values()) == 0:
                return
            
            plt.figure(figsize=(8, 6))
            labels = list(distribution.keys())
            sizes = list(distribution.values())
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Sentiment Distribution - {date_str}')
            plt.axis('equal')
            
            plt.savefig(viz_dir / f'sentiment_distribution_{date_str}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating sentiment distribution chart: {e}")
    
    def _create_sentiment_price_scatter(self, report: Dict[str, Any], viz_dir: Path, date_str: str):
        """Create sentiment vs price change scatter plot."""
        try:
            ticker_analyses = report['ticker_analyses']
            
            tickers = []
            sentiments = []
            price_changes = []
            
            for ticker, analysis in ticker_analyses.items():
                if analysis['tweet_count'] > 0:  # Only include tickers with tweets
                    tickers.append(ticker)
                    sentiments.append(analysis['avg_sentiment'])
                    price_changes.append(analysis['price_change_percent'])
            
            if not tickers:
                return
            
            plt.figure(figsize=(10, 6))
            plt.scatter(sentiments, price_changes, alpha=0.7, s=100)
            
            # Add ticker labels
            for i, ticker in enumerate(tickers):
                plt.annotate(ticker, (sentiments[i], price_changes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Average Sentiment Score')
            plt.ylabel('Price Change (%)')
            plt.title(f'Sentiment vs Price Change - {date_str}')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            plt.savefig(viz_dir / f'sentiment_price_scatter_{date_str}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating sentiment-price scatter plot: {e}")
    
    def _create_ticker_performance_chart(self, report: Dict[str, Any], viz_dir: Path, date_str: str):
        """Create ticker performance bar chart."""
        try:
            ticker_analyses = report['ticker_analyses']
            
            tickers = list(ticker_analyses.keys())
            price_changes = [analysis['price_change_percent'] for analysis in ticker_analyses.values()]
            
            plt.figure(figsize=(12, 6))
            colors = ['#2ecc71' if pc > 0 else '#e74c3c' if pc < 0 else '#95a5a6' for pc in price_changes]
            
            bars = plt.bar(tickers, price_changes, color=colors, alpha=0.7)
            
            plt.xlabel('Ticker')
            plt.ylabel('Price Change (%)')
            plt.title(f'Ticker Performance - {date_str}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, price_changes):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                        f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'ticker_performance_{date_str}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating ticker performance chart: {e}")
    
    def _convert_datetime_to_string(self, obj):
        """Recursively convert datetime objects to strings for JSON serialization."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_datetime_to_string(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_to_string(item) for item in obj]
        else:
            return obj
    
    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest daily report."""
        try:
            report_files = list(self.reports_dir.glob("daily_report_*.json"))
            
            if not report_files:
                return None
            
            # Get the most recent file
            latest_file = max(report_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                report = json.load(f)
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting latest report: {e}")
            return None