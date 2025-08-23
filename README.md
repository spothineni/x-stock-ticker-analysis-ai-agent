# 📈 Stock Sentiment Analyzer

A comprehensive system that analyzes Twitter/X tweets related to stock tickers to generate daily knowledge about stock ranges and forecasts. This project combines social media sentiment analysis with financial data analysis to provide insights and predictions for stock movements.

## 🚀 Features

- **Real-time Tweet Collection**: Collects tweets mentioning stock tickers using Twitter/X API
- **Advanced Sentiment Analysis**: Multi-model sentiment analysis optimized for financial content
- **Stock Data Integration**: Fetches real-time and historical stock data using Yahoo Finance
- **Technical Analysis**: Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Machine Learning Predictions**: Uses Random Forest and Linear Regression for price forecasting
- **Daily Reports**: Generates comprehensive daily analysis reports with visualizations
- **Web Dashboard**: Interactive web interface for viewing results and managing the system
- **Automated Scheduling**: Daily automated data collection and analysis
- **RESTful API**: Complete API for integration with other systems

## 🏗️ Architecture

```
├── src/
│   ├── data_collection/     # Twitter and stock data collection
│   ├── sentiment_analysis/  # Tweet processing and sentiment analysis
│   ├── stock_analysis/      # Technical analysis and correlation studies
│   ├── forecasting/         # Daily forecasting engine and report generation
│   ├── web_interface/       # Flask web application and API
│   └── utils/              # Configuration, database, and logging utilities
├── config/                 # Configuration files
├── data/                   # Data storage (raw, processed, reports)
├── logs/                   # Application logs
└── models/                 # Trained ML models
```

## 📋 Prerequisites

- Python 3.11+
- Twitter/X API credentials (Bearer Token)
- Internet connection for stock data APIs

## 🛠️ Installation

### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd openhands-test
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

4. **Run the application**:
   ```bash
   # Start web server
   python main.py web
   
   # Or run a single forecast
   python main.py forecast
   ```

### Option 2: Docker Installation

1. **Using Docker Compose** (Recommended):
   ```bash
   # Copy environment file
   cp .env.example .env
   # Edit .env with your credentials
   
   # Start services
   docker-compose up -d
   ```

2. **Using Docker directly**:
   ```bash
   # Build image
   docker build -t stock-sentiment-analyzer .
   
   # Run container
   docker run -p 12000:12000 -v $(pwd)/.env:/app/.env stock-sentiment-analyzer
   ```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Twitter/X API Credentials
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
FLASK_HOST=0.0.0.0
FLASK_PORT=12000
```

### Stock Tickers Configuration

Edit `config/config.yaml` to modify the list of tracked stock tickers:

```yaml
data_collection:
  stock_tickers:
    - "AAPL"
    - "GOOGL"
    - "MSFT"
    - "TSLA"
    # Add more tickers as needed
```

## 🚀 Usage

### Web Dashboard

1. Start the web server:
   ```bash
   python main.py web
   ```

2. Open your browser to `http://localhost:12000`

3. Use the dashboard to:
   - View real-time market sentiment
   - Monitor stock predictions
   - Trigger manual data collection
   - Generate reports

### Command Line Interface

```bash
# Run daily forecast
python main.py forecast

# Generate daily report
python main.py report

# Start web server
python main.py web

# Set up scheduler instructions
python main.py scheduler
```

### Automated Scheduling

```bash
# Run scheduler (keeps running and executes daily tasks)
python scheduler.py

# Run forecast immediately
python scheduler.py --run-now

# Run data collection only
python scheduler.py --run-now --data-only
```

### API Endpoints

The system provides a comprehensive REST API:

- `GET /api/status` - System status
- `GET /api/tickers` - List of configured tickers
- `GET /api/latest-report` - Latest daily report
- `GET /api/ticker/{ticker}/analysis` - Detailed ticker analysis
- `GET /api/ticker/{ticker}/prediction` - Price predictions
- `GET /api/ticker/{ticker}/sentiment` - Sentiment analysis
- `POST /api/run-forecast` - Trigger manual forecast
- `POST /api/generate-report` - Generate new report
- `POST /api/collect-data` - Collect new data

## 📊 Data Flow

1. **Data Collection**: 
   - Collects tweets mentioning configured stock tickers
   - Fetches current and historical stock price data

2. **Sentiment Analysis**:
   - Processes tweets using multiple sentiment models
   - Applies financial-specific sentiment rules
   - Calculates confidence scores

3. **Technical Analysis**:
   - Computes technical indicators
   - Analyzes price patterns and trends
   - Calculates support/resistance levels

4. **Prediction**:
   - Combines sentiment and technical data
   - Uses machine learning models for forecasting
   - Generates price range predictions with confidence intervals

5. **Reporting**:
   - Creates comprehensive daily reports
   - Generates visualizations
   - Provides actionable insights

## 🔧 Customization

### Adding New Stock Tickers

1. Edit `config/config.yaml`:
   ```yaml
   data_collection:
     stock_tickers:
       - "YOUR_TICKER"
   ```

2. Restart the application

### Modifying Sentiment Analysis

- Edit `src/sentiment_analysis/sentiment_analyzer.py`
- Adjust model weights in the `model_weights` dictionary
- Add custom financial terms to the lexicon

### Customizing Predictions

- Modify `src/stock_analysis/predictor.py`
- Adjust feature selection in `feature_columns`
- Tune model parameters

## 📈 Performance Optimization

### For High-Volume Usage

1. **Database Optimization**:
   - Consider using PostgreSQL instead of SQLite
   - Add database indexes for frequently queried columns

2. **Caching**:
   - Implement Redis for caching API responses
   - Cache sentiment analysis results

3. **Scaling**:
   - Use Celery for background task processing
   - Deploy multiple instances behind a load balancer

## 🐛 Troubleshooting

### Common Issues

1. **Twitter API Rate Limits**:
   - The system includes rate limiting handling
   - Increase delays in `config/config.yaml` if needed

2. **No Data Collected**:
   - Check Twitter API credentials
   - Verify internet connection
   - Check logs in `logs/` directory

3. **Prediction Errors**:
   - Ensure sufficient historical data (minimum 20 days)
   - Check for missing sentiment data

4. **Web Interface Not Loading**:
   - Verify port 12000 is not in use
   - Check firewall settings
   - Review application logs

### Logging

Logs are stored in the `logs/` directory:
- `stock_sentiment.log` - Main application log
- Logs rotate daily and are compressed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Twitter/X API for social media data
- Yahoo Finance for stock market data
- Various open-source libraries for sentiment analysis and machine learning

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review application logs
- Open an issue on GitHub

---

**Disclaimer**: This tool is for educational and research purposes only. Do not use it as the sole basis for investment decisions. Always consult with financial professionals and do your own research before making investment decisions.
