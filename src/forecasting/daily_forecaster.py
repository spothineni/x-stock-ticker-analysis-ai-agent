"""Daily forecasting engine that combines all analysis components."""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
from ..data_collection import TwitterCollector, StockCollector
from ..sentiment_analysis import SentimentAnalyzer
from ..stock_analysis import StockAnalyzer, StockPredictor
from ..utils.database import db_manager
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DailyForecaster:
    """Main forecasting engine that orchestrates daily analysis."""
    
    def __init__(self):
        """Initialize daily forecaster with all components."""
        self.twitter_collector = TwitterCollector()
        self.stock_collector = StockCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stock_analyzer = StockAnalyzer()
        self.stock_predictor = StockPredictor()
        
        # Create output directories
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Daily forecaster initialized")
    
    def collect_daily_data(self) -> Dict[str, Any]:
        """Collect all daily data (tweets and stock prices)."""
        logger.info("Starting daily data collection")
        
        collection_results = {
            'timestamp': datetime.utcnow(),
            'tweets_collected': 0,
            'stock_data_collected': 0,
            'errors': []
        }
        
        try:
            # Collect tweets
            if self.twitter_collector.client:
                logger.info("Collecting tweets...")
                all_tweets = self.twitter_collector.collect_tweets_for_all_tickers()
                
                # Flatten tweets and save to database
                all_tweet_records = []
                for ticker, tweets in all_tweets.items():
                    all_tweet_records.extend(tweets)
                
                if all_tweet_records:
                    saved_tweets = db_manager.save_tweets(all_tweet_records)
                    collection_results['tweets_collected'] = saved_tweets
                    logger.info(f"Saved {saved_tweets} new tweets")
                else:
                    logger.warning("No tweets collected")
            else:
                logger.warning("Twitter API not available, skipping tweet collection")
                collection_results['errors'].append("Twitter API not available")
            
            # Collect stock data
            logger.info("Collecting stock data...")
            stock_data = self.stock_collector.collect_daily_data(days_back=1)
            
            # Flatten stock data and save to database
            all_stock_records = []
            for ticker, records in stock_data.items():
                all_stock_records.extend(records)
            
            if all_stock_records:
                saved_stock_data = db_manager.save_stock_data(all_stock_records)
                collection_results['stock_data_collected'] = saved_stock_data
                logger.info(f"Saved {saved_stock_data} new stock data records")
            else:
                logger.warning("No stock data collected")
                collection_results['errors'].append("No stock data collected")
            
        except Exception as e:
            error_msg = f"Error during data collection: {e}"
            logger.error(error_msg)
            collection_results['errors'].append(error_msg)
        
        return collection_results
    
    def analyze_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment for all collected tweets."""
        logger.info("Starting sentiment analysis")
        
        analysis_results = {
            'timestamp': datetime.utcnow(),
            'tweets_analyzed': 0,
            'sentiment_records_saved': 0,
            'ticker_summaries': {},
            'errors': []
        }
        
        try:
            for ticker in config.stock_tickers:
                logger.info(f"Analyzing sentiment for {ticker}")
                
                # Get recent tweets for this ticker
                tweets = db_manager.get_tweets_by_ticker(ticker, days_back=1)
                
                if not tweets:
                    logger.info(f"No tweets found for {ticker}")
                    continue
                
                # Convert to format expected by sentiment analyzer
                tweet_data = [{
                    'tweet_id': tweet.tweet_id,
                    'text': tweet.text,
                    'ticker': tweet.ticker
                } for tweet in tweets]
                
                # Analyze sentiment
                sentiment_results = self.sentiment_analyzer.analyze_tweet_batch(tweet_data)
                
                if sentiment_results:
                    # Save sentiment analysis results
                    saved_count = db_manager.save_sentiment_analysis(sentiment_results)
                    analysis_results['sentiment_records_saved'] += saved_count
                    analysis_results['tweets_analyzed'] += len(tweet_data)
                    
                    # Generate ticker summary
                    ticker_summary = self.sentiment_analyzer.get_ticker_sentiment_summary(
                        sentiment_results, ticker
                    )
                    analysis_results['ticker_summaries'][ticker] = ticker_summary
                    
                    logger.info(f"Analyzed {len(tweet_data)} tweets for {ticker}")
        
        except Exception as e:
            error_msg = f"Error during sentiment analysis: {e}"
            logger.error(error_msg)
            analysis_results['errors'].append(error_msg)
        
        return analysis_results
    
    def generate_stock_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive stock analysis for all tickers."""
        logger.info("Starting stock analysis")
        
        stock_analysis_results = {
            'timestamp': datetime.utcnow(),
            'tickers_analyzed': 0,
            'analyses': {},
            'errors': []
        }
        
        try:
            for ticker in config.stock_tickers:
                logger.info(f"Generating stock analysis for {ticker}")
                
                analysis = self.stock_analyzer.generate_daily_analysis(ticker)
                
                if analysis:
                    stock_analysis_results['analyses'][ticker] = analysis
                    stock_analysis_results['tickers_analyzed'] += 1
                    
                    # Save to database
                    daily_analysis_record = {
                        'ticker': ticker,
                        'analysis_date': analysis['analysis_date'],
                        'tweet_count': analysis['tweet_count'],
                        'avg_sentiment': analysis['avg_sentiment'],
                        'sentiment_std': analysis['sentiment_std'],
                        'price_change_percent': analysis['price_change_percent'],
                        'volume_change_percent': analysis['volume_change_percent']
                    }
                    
                    db_manager.save_daily_analysis(daily_analysis_record)
                    
                else:
                    error_msg = f"Failed to generate analysis for {ticker}"
                    logger.warning(error_msg)
                    stock_analysis_results['errors'].append(error_msg)
        
        except Exception as e:
            error_msg = f"Error during stock analysis: {e}"
            logger.error(error_msg)
            stock_analysis_results['errors'].append(error_msg)
        
        return stock_analysis_results
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate price predictions for all tickers."""
        logger.info("Starting price predictions")
        
        prediction_results = {
            'timestamp': datetime.utcnow(),
            'predictions_generated': 0,
            'predictions': {},
            'errors': []
        }
        
        try:
            # Generate predictions using Random Forest model
            predictions = self.stock_predictor.batch_predict_all_tickers('random_forest')
            
            for ticker, prediction in predictions.items():
                if prediction.get('success', False):
                    prediction_results['predictions'][ticker] = prediction
                    prediction_results['predictions_generated'] += 1
                    
                    # Update daily analysis record with predictions
                    try:
                        session = db_manager.get_session()
                        from ..utils.database import DailyAnalysis
                        
                        # Find today's analysis record
                        today = datetime.utcnow().date()
                        analysis_record = session.query(DailyAnalysis).filter(
                            DailyAnalysis.ticker == ticker,
                            DailyAnalysis.analysis_date >= today
                        ).first()
                        
                        if analysis_record:
                            analysis_record.predicted_range_low = prediction['predicted_range_low']
                            analysis_record.predicted_range_high = prediction['predicted_range_high']
                            analysis_record.prediction_confidence = prediction['prediction_confidence']
                            session.commit()
                        
                        session.close()
                        
                    except Exception as e:
                        logger.error(f"Error updating analysis record for {ticker}: {e}")
                
                else:
                    error_msg = f"Failed to generate prediction for {ticker}: {prediction.get('error', 'Unknown error')}"
                    logger.warning(error_msg)
                    prediction_results['errors'].append(error_msg)
        
        except Exception as e:
            error_msg = f"Error during prediction generation: {e}"
            logger.error(error_msg)
            prediction_results['errors'].append(error_msg)
        
        return prediction_results
    
    def run_daily_forecast(self) -> Dict[str, Any]:
        """Run complete daily forecasting pipeline."""
        logger.info("Starting daily forecast pipeline")
        
        pipeline_results = {
            'started_at': datetime.utcnow(),
            'completed_at': None,
            'success': False,
            'stages': {},
            'summary': {},
            'errors': []
        }
        
        try:
            # Stage 1: Data Collection
            logger.info("Stage 1: Data Collection")
            collection_results = self.collect_daily_data()
            pipeline_results['stages']['data_collection'] = collection_results
            
            # Stage 2: Sentiment Analysis
            logger.info("Stage 2: Sentiment Analysis")
            sentiment_results = self.analyze_sentiment()
            pipeline_results['stages']['sentiment_analysis'] = sentiment_results
            
            # Stage 3: Stock Analysis
            logger.info("Stage 3: Stock Analysis")
            stock_analysis_results = self.generate_stock_analysis()
            pipeline_results['stages']['stock_analysis'] = stock_analysis_results
            
            # Stage 4: Predictions
            logger.info("Stage 4: Price Predictions")
            prediction_results = self.generate_predictions()
            pipeline_results['stages']['predictions'] = prediction_results
            
            # Generate summary
            pipeline_results['summary'] = {
                'tweets_collected': collection_results['tweets_collected'],
                'stock_data_collected': collection_results['stock_data_collected'],
                'tweets_analyzed': sentiment_results['tweets_analyzed'],
                'tickers_analyzed': stock_analysis_results['tickers_analyzed'],
                'predictions_generated': prediction_results['predictions_generated'],
                'total_errors': (
                    len(collection_results['errors']) +
                    len(sentiment_results['errors']) +
                    len(stock_analysis_results['errors']) +
                    len(prediction_results['errors'])
                )
            }
            
            # Collect all errors
            all_errors = []
            all_errors.extend(collection_results['errors'])
            all_errors.extend(sentiment_results['errors'])
            all_errors.extend(stock_analysis_results['errors'])
            all_errors.extend(prediction_results['errors'])
            pipeline_results['errors'] = all_errors
            
            # Determine success
            pipeline_results['success'] = (
                pipeline_results['summary']['total_errors'] == 0 or
                (pipeline_results['summary']['predictions_generated'] > 0 and
                 pipeline_results['summary']['tweets_analyzed'] > 0)
            )
            
            pipeline_results['completed_at'] = datetime.utcnow()
            
            # Save pipeline results
            self._save_pipeline_results(pipeline_results)
            
            logger.info(f"Daily forecast pipeline completed. Success: {pipeline_results['success']}")
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Critical error in daily forecast pipeline: {e}"
            logger.error(error_msg)
            pipeline_results['errors'].append(error_msg)
            pipeline_results['completed_at'] = datetime.utcnow()
            pipeline_results['success'] = False
            
            return pipeline_results
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file."""
        try:
            timestamp = results['started_at'].strftime('%Y%m%d_%H%M%S')
            filename = f"daily_forecast_{timestamp}.json"
            filepath = self.reports_dir / filename
            
            # Convert datetime objects to strings for JSON serialization
            json_results = self._convert_datetime_to_string(results)
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
    
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
    
    def get_latest_forecast_summary(self) -> Dict[str, Any]:
        """Get summary of the latest forecast results."""
        try:
            # Find the most recent pipeline results file
            result_files = list(self.reports_dir.glob("daily_forecast_*.json"))
            
            if not result_files:
                return {
                    'success': False,
                    'error': 'No forecast results found'
                }
            
            # Get the most recent file
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            return {
                'success': True,
                'forecast_date': results.get('started_at'),
                'summary': results.get('summary', {}),
                'pipeline_success': results.get('success', False),
                'error_count': len(results.get('errors', [])),
                'file_path': str(latest_file)
            }
            
        except Exception as e:
            logger.error(f"Error getting latest forecast summary: {e}")
            return {
                'success': False,
                'error': str(e)
            }