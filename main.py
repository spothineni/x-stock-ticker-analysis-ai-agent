"""Main application entry point for the Stock Sentiment Analyzer."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.web_interface import create_app
from src.forecasting import DailyForecaster, ReportGenerator
from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_web_server():
    """Run the web server."""
    logger.info("Starting Stock Sentiment Analyzer web server")
    
    app = create_app()
    
    web_config = config.web_config
    
    logger.info(f"Web server starting on {web_config['host']}:{web_config['port']}")
    
    app.run(
        host=web_config['host'],
        port=web_config['port'],
        debug=web_config['debug']
    )

def run_forecast():
    """Run a single forecast cycle."""
    logger.info("Running daily forecast")
    
    forecaster = DailyForecaster()
    results = forecaster.run_daily_forecast()
    
    if results['success']:
        logger.info("Forecast completed successfully")
        print("✅ Forecast completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Tweets collected: {results['summary'].get('tweets_collected', 0)}")
        print(f"   - Stock data collected: {results['summary'].get('stock_data_collected', 0)}")
        print(f"   - Tweets analyzed: {results['summary'].get('tweets_analyzed', 0)}")
        print(f"   - Tickers analyzed: {results['summary'].get('tickers_analyzed', 0)}")
        print(f"   - Predictions generated: {results['summary'].get('predictions_generated', 0)}")
        
        if results['errors']:
            print(f"⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"   - {error}")
    else:
        logger.error("Forecast failed")
        print("❌ Forecast failed!")
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"   - {error}")
        return 1
    
    return 0

def generate_report():
    """Generate a daily report."""
    logger.info("Generating daily report")
    
    report_generator = ReportGenerator()
    report = report_generator.generate_daily_report()
    
    if report:
        logger.info("Report generated successfully")
        print("✅ Report generated successfully!")
        print(f"📊 Report Summary:")
        print(f"   - Report date: {report['report_date']}")
        print(f"   - Market trend: {report['market_summary'].get('market_trend', 'unknown')}")
        print(f"   - Tickers analyzed: {report['market_summary'].get('analyzed_tickers', 0)}")
        print(f"   - Total tweets: {report['market_summary'].get('total_tweets', 0)}")
        print(f"   - Average sentiment: {report['market_summary'].get('avg_sentiment', 0):.3f}")
        
        # Show key insights
        if report.get('key_insights'):
            print("🔍 Key Insights:")
            for insight in report['key_insights'][:3]:  # Show first 3 insights
                print(f"   - {insight}")
    else:
        logger.error("Report generation failed")
        print("❌ Report generation failed!")
        return 1
    
    return 0

def setup_scheduler():
    """Set up automated scheduling (placeholder for production deployment)."""
    print("📅 Scheduler Setup Instructions:")
    print()
    print("To set up automated daily forecasting, you can use one of these methods:")
    print()
    print("1. Cron Job (Linux/Mac):")
    print("   Add this line to your crontab (crontab -e):")
    print(f"   0 9 * * * cd {Path(__file__).parent} && python main.py forecast")
    print()
    print("2. Windows Task Scheduler:")
    print("   Create a daily task that runs:")
    print(f"   Program: python")
    print(f"   Arguments: {Path(__file__).absolute()} forecast")
    print(f"   Start in: {Path(__file__).parent}")
    print()
    print("3. Docker with cron:")
    print("   Use the provided Dockerfile and add a cron job inside the container")
    print()
    print("4. Cloud scheduler (AWS EventBridge, Google Cloud Scheduler, etc.):")
    print("   Set up a daily trigger to call the /api/run-forecast endpoint")
    print()
    print("The recommended time is 9:00 AM UTC (after market open) for daily analysis.")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Stock Sentiment Analyzer')
    parser.add_argument('command', choices=['web', 'forecast', 'report', 'scheduler'], 
                       help='Command to run')
    parser.add_argument('--port', type=int, default=None, 
                       help='Port for web server (overrides config)')
    parser.add_argument('--host', type=str, default=None, 
                       help='Host for web server (overrides config)')
    
    args = parser.parse_args()
    
    # Override config if command line args provided
    if args.port:
        config.web_config['port'] = args.port
    if args.host:
        config.web_config['host'] = args.host
    
    try:
        if args.command == 'web':
            run_web_server()
        elif args.command == 'forecast':
            return run_forecast()
        elif args.command == 'report':
            return generate_report()
        elif args.command == 'scheduler':
            setup_scheduler()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())