"""Automated scheduler for daily forecasting tasks."""

import schedule
import time
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.forecasting import DailyForecaster, ReportGenerator
from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AutomatedScheduler:
    """Handles automated scheduling of daily tasks."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.forecaster = DailyForecaster()
        self.report_generator = ReportGenerator()
        logger.info("Automated scheduler initialized")
    
    def run_daily_forecast_job(self):
        """Job function for daily forecast."""
        logger.info("Starting scheduled daily forecast")
        
        try:
            # Run the forecast
            results = self.forecaster.run_daily_forecast()
            
            if results['success']:
                logger.info("Scheduled forecast completed successfully")
                
                # Generate report after successful forecast
                logger.info("Generating daily report")
                report = self.report_generator.generate_daily_report()
                
                if report:
                    logger.info("Daily report generated successfully")
                else:
                    logger.warning("Daily report generation failed")
                
            else:
                logger.error("Scheduled forecast failed")
                logger.error(f"Errors: {results.get('errors', [])}")
                
        except Exception as e:
            logger.error(f"Error in scheduled forecast job: {e}")
    
    def run_data_collection_job(self):
        """Job function for data collection only."""
        logger.info("Starting scheduled data collection")
        
        try:
            results = self.forecaster.collect_daily_data()
            
            if results['tweets_collected'] > 0 or results['stock_data_collected'] > 0:
                logger.info(f"Data collection completed: {results['tweets_collected']} tweets, {results['stock_data_collected']} stock records")
            else:
                logger.warning("No data collected")
                
        except Exception as e:
            logger.error(f"Error in data collection job: {e}")
    
    def setup_schedule(self):
        """Set up the scheduled jobs."""
        # Get scheduling configuration
        daily_time = config.get('scheduler.daily_analysis_time', '09:00')
        timezone = config.get('scheduler.timezone', 'UTC')
        
        logger.info(f"Setting up daily forecast at {daily_time} {timezone}")
        
        # Schedule daily forecast (includes data collection, analysis, and reporting)
        schedule.every().day.at(daily_time).do(self.run_daily_forecast_job)
        
        # Optional: Schedule additional data collection at market open (if different from analysis time)
        # schedule.every().day.at("09:30").do(self.run_data_collection_job)
        
        # Optional: Schedule evening data collection for after-hours sentiment
        # schedule.every().day.at("18:00").do(self.run_data_collection_job)
        
        logger.info("Scheduled jobs configured")
    
    def run_scheduler(self):
        """Run the scheduler loop."""
        logger.info("Starting scheduler loop")
        print(f"🕐 Scheduler started. Daily forecast scheduled for {config.get('scheduler.daily_analysis_time', '09:00')} UTC")
        print("Press Ctrl+C to stop the scheduler")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            print("\n👋 Scheduler stopped")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            print(f"❌ Scheduler error: {e}")

def main():
    """Main function for running the scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Sentiment Analyzer Scheduler')
    parser.add_argument('--run-now', action='store_true', 
                       help='Run forecast immediately instead of scheduling')
    parser.add_argument('--data-only', action='store_true',
                       help='Run data collection only')
    
    args = parser.parse_args()
    
    scheduler = AutomatedScheduler()
    
    if args.run_now:
        if args.data_only:
            print("🔄 Running data collection...")
            scheduler.run_data_collection_job()
        else:
            print("🔄 Running daily forecast...")
            scheduler.run_daily_forecast_job()
        print("✅ Job completed")
    else:
        scheduler.setup_schedule()
        scheduler.run_scheduler()

if __name__ == '__main__':
    main()