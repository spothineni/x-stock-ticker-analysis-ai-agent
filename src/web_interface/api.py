"""API endpoints for the stock sentiment analyzer."""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from ..forecasting import DailyForecaster, ReportGenerator
from ..data_collection import TwitterCollector, StockCollector
from ..sentiment_analysis import SentimentAnalyzer
from ..stock_analysis import StockAnalyzer, StockPredictor
from ..utils.database import db_manager
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

api_bp = Blueprint('api', __name__)

# Initialize components
daily_forecaster = DailyForecaster()
report_generator = ReportGenerator()
twitter_collector = TwitterCollector()
stock_collector = StockCollector()
sentiment_analyzer = SentimentAnalyzer()
stock_analyzer = StockAnalyzer()
stock_predictor = StockPredictor()

@api_bp.route('/status')
def get_status():
    """Get system status."""
    try:
        # Check database connection
        session = db_manager.get_session()
        session.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check Twitter API
    twitter_status = "available" if twitter_collector.client else "not configured"
    
    return jsonify({
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat(),
        'database': db_status,
        'twitter_api': twitter_status,
        'configured_tickers': config.stock_tickers
    })

@api_bp.route('/tickers')
def get_tickers():
    """Get list of configured stock tickers."""
    return jsonify({
        'tickers': config.stock_tickers,
        'count': len(config.stock_tickers)
    })

@api_bp.route('/latest-report')
def get_latest_report():
    """Get the latest daily report."""
    try:
        report = report_generator.get_latest_report()
        
        if report:
            return jsonify({
                'success': True,
                'report': report
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No reports available'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting latest report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/forecast-summary')
def get_forecast_summary():
    """Get summary of the latest forecast."""
    try:
        summary = daily_forecaster.get_latest_forecast_summary()
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting forecast summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/ticker/<ticker>/analysis')
def get_ticker_analysis(ticker):
    """Get detailed analysis for a specific ticker."""
    try:
        if ticker not in config.stock_tickers:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} not configured'
            }), 400
        
        # Get stock analysis
        analysis = stock_analyzer.generate_daily_analysis(ticker)
        
        if analysis:
            return jsonify({
                'success': True,
                'analysis': analysis
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No analysis available for {ticker}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting analysis for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/ticker/<ticker>/prediction')
def get_ticker_prediction(ticker):
    """Get price prediction for a specific ticker."""
    try:
        if ticker not in config.stock_tickers:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} not configured'
            }), 400
        
        # Get prediction
        prediction = stock_predictor.predict_price_range(ticker)
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error getting prediction for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/ticker/<ticker>/sentiment')
def get_ticker_sentiment(ticker):
    """Get sentiment analysis for a specific ticker."""
    try:
        if ticker not in config.stock_tickers:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} not configured'
            }), 400
        
        # Get recent tweets
        tweets = db_manager.get_tweets_by_ticker(ticker, days_back=7)
        
        if not tweets:
            return jsonify({
                'success': True,
                'ticker': ticker,
                'tweet_count': 0,
                'sentiment_summary': {
                    'avg_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                }
            })
        
        # Get sentiment analysis
        session = db_manager.get_session()
        from ..utils.database import SentimentAnalysis
        
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        sentiment_data = session.query(SentimentAnalysis).filter(
            SentimentAnalysis.ticker == ticker,
            SentimentAnalysis.analyzed_at >= cutoff_date
        ).all()
        
        session.close()
        
        if sentiment_data:
            # Calculate summary
            sentiment_results = [{
                'sentiment_score': s.sentiment_score,
                'sentiment_label': s.sentiment_label,
                'confidence': s.confidence
            } for s in sentiment_data]
            
            summary = sentiment_analyzer.get_ticker_sentiment_summary(sentiment_results, ticker)
        else:
            summary = {
                'ticker': ticker,
                'tweet_count': len(tweets),
                'avg_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'tweet_count': len(tweets),
            'sentiment_summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/run-forecast', methods=['POST'])
def run_forecast():
    """Trigger a manual forecast run."""
    try:
        logger.info("Manual forecast triggered via API")
        
        # Run the forecast pipeline
        results = daily_forecaster.run_daily_forecast()
        
        return jsonify({
            'success': results['success'],
            'message': 'Forecast completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error running manual forecast: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate a new daily report."""
    try:
        logger.info("Manual report generation triggered via API")
        
        # Generate report
        report = report_generator.generate_daily_report()
        
        return jsonify({
            'success': True,
            'message': 'Report generated successfully',
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/collect-data', methods=['POST'])
def collect_data():
    """Trigger manual data collection."""
    try:
        logger.info("Manual data collection triggered via API")
        
        # Collect data
        results = daily_forecaster.collect_daily_data()
        
        return jsonify({
            'success': True,
            'message': 'Data collection completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/market-summary')
def get_market_summary():
    """Get overall market summary."""
    try:
        # Get market summary from stock collector
        market_summary = stock_collector.get_market_summary()
        
        # Get sentiment overview
        session = db_manager.get_session()
        from ..utils.database import SentimentAnalysis
        
        # Get sentiment data from last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_sentiments = session.query(SentimentAnalysis).filter(
            SentimentAnalysis.analyzed_at >= cutoff_time
        ).all()
        
        session.close()
        
        # Calculate sentiment summary
        if recent_sentiments:
            avg_sentiment = sum(s.sentiment_score for s in recent_sentiments) / len(recent_sentiments)
            positive_count = sum(1 for s in recent_sentiments if s.sentiment_label == 'positive')
            negative_count = sum(1 for s in recent_sentiments if s.sentiment_label == 'negative')
            neutral_count = sum(1 for s in recent_sentiments if s.sentiment_label == 'neutral')
        else:
            avg_sentiment = 0.0
            positive_count = negative_count = neutral_count = 0
        
        market_summary['sentiment_overview'] = {
            'avg_sentiment': avg_sentiment,
            'total_tweets_analyzed': len(recent_sentiments),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }
        
        return jsonify({
            'success': True,
            'market_summary': market_summary
        })
        
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/historical-data/<ticker>')
def get_historical_data(ticker):
    """Get historical data for a ticker."""
    try:
        if ticker not in config.stock_tickers:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} not configured'
            }), 400
        
        days_back = request.args.get('days', 30, type=int)
        days_back = min(days_back, 365)  # Limit to 1 year
        
        # Get stock data
        stock_data = db_manager.get_stock_data_by_ticker(ticker, days_back)
        
        # Get sentiment data
        session = db_manager.get_session()
        from ..utils.database import SentimentAnalysis
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        sentiment_data = session.query(SentimentAnalysis).filter(
            SentimentAnalysis.ticker == ticker,
            SentimentAnalysis.analyzed_at >= cutoff_date
        ).all()
        
        session.close()
        
        # Format data
        historical_data = {
            'ticker': ticker,
            'stock_data': [{
                'date': s.date.isoformat(),
                'open': s.open_price,
                'high': s.high_price,
                'low': s.low_price,
                'close': s.close_price,
                'volume': s.volume
            } for s in stock_data],
            'sentiment_data': [{
                'date': s.analyzed_at.date().isoformat(),
                'sentiment_score': s.sentiment_score,
                'sentiment_label': s.sentiment_label,
                'confidence': s.confidence
            } for s in sentiment_data]
        }
        
        return jsonify({
            'success': True,
            'data': historical_data
        })
        
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500