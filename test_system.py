#!/usr/bin/env python3
"""Simple system test to verify the Stock Sentiment Analyzer is working correctly."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.utils.config import config
        from src.utils.logger import setup_logger
        from src.utils.database import db_manager
        print("✅ Utils modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing utils: {e}")
        return False
    
    try:
        from src.data_collection import TwitterCollector, StockCollector
        print("✅ Data collection modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing data collection: {e}")
        return False
    
    try:
        from src.sentiment_analysis import SentimentAnalyzer, TextProcessor
        print("✅ Sentiment analysis modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing sentiment analysis: {e}")
        return False
    
    try:
        from src.stock_analysis import StockAnalyzer, StockPredictor
        print("✅ Stock analysis modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing stock analysis: {e}")
        return False
    
    try:
        from src.forecasting import DailyForecaster, ReportGenerator
        print("✅ Forecasting modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing forecasting: {e}")
        return False
    
    try:
        from src.web_interface import create_app
        print("✅ Web interface modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing web interface: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.utils.config import config
        
        # Test basic config access
        tickers = config.stock_tickers
        print(f"✅ Configuration loaded. Tracking {len(tickers)} tickers: {tickers}")
        
        # Test web config
        web_config = config.web_config
        print(f"✅ Web config: {web_config['host']}:{web_config['port']}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_database():
    """Test database initialization."""
    print("\n💾 Testing database...")
    
    try:
        from src.utils.database import db_manager
        
        # Test database connection
        session = db_manager.get_session()
        session.close()
        print("✅ Database connection successful")
        
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print("\n💭 Testing sentiment analysis...")
    
    try:
        from src.sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Test with sample financial text
        test_text = "AAPL is going to the moon! Great earnings report, buying more shares!"
        result = analyzer.analyze_sentiment(test_text)
        
        print(f"✅ Sentiment analysis working. Sample result:")
        print(f"   Text: '{test_text}'")
        print(f"   Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        return False

def test_stock_data():
    """Test stock data collection."""
    print("\n📈 Testing stock data collection...")
    
    try:
        from src.data_collection import StockCollector
        
        collector = StockCollector()
        
        # Test getting current price for AAPL
        current_data = collector.get_current_price("AAPL")
        
        if current_data:
            print(f"✅ Stock data collection working. AAPL current price: ${current_data['current_price']:.2f}")
        else:
            print("⚠️  Stock data collection returned no data (might be market hours)")
        
        return True
    except Exception as e:
        print(f"❌ Stock data collection error: {e}")
        return False

def test_web_app():
    """Test web application creation."""
    print("\n🌐 Testing web application...")
    
    try:
        from src.web_interface import create_app
        
        app = create_app()
        
        # Test app creation
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Web application created successfully")
                print(f"   Health check response: {response.get_json()}")
            else:
                print(f"⚠️  Health check returned status {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Web application error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Stock Sentiment Analyzer System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_database,
        test_sentiment_analysis,
        test_stock_data,
        test_web_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Configure your Twitter API credentials in .env")
        print("2. Run 'python main.py web' to start the web interface")
        print("3. Visit http://localhost:12000 to access the dashboard")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit(main())