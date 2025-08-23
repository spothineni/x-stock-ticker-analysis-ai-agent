"""Stock market data collection module."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import yfinance as yf
import pandas as pd
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class StockCollector:
    """Collects stock market data for analysis."""
    
    def __init__(self):
        """Initialize stock data collector."""
        self.tickers = config.stock_tickers
        logger.info(f"Stock collector initialized for {len(self.tickers)} tickers")
    
    def get_stock_data(self, ticker: str, period: str = "1d", interval: str = "1m") -> Optional[pd.DataFrame]:
        """Get stock data for a specific ticker."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None
            
            # Reset index to get datetime as a column
            hist.reset_index(inplace=True)
            
            # Add ticker column
            hist['ticker'] = ticker
            
            logger.info(f"Retrieved {len(hist)} data points for {ticker}")
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get current price and basic info for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_data = {
                'ticker': ticker,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'previous_close': info.get('previousClose'),
                'open': info.get('open', info.get('regularMarketOpen')),
                'day_high': info.get('dayHigh', info.get('regularMarketDayHigh')),
                'day_low': info.get('dayLow', info.get('regularMarketDayLow')),
                'volume': info.get('volume', info.get('regularMarketVolume')),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': datetime.utcnow()
            }
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None
    
    def collect_daily_data(self, days_back: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """Collect daily stock data for all configured tickers."""
        all_stock_data = {}
        
        for ticker in self.tickers:
            logger.info(f"Collecting stock data for {ticker}")
            
            # Get historical data
            df = self.get_stock_data(ticker, period=f"{days_back}d", interval="1d")
            
            if df is not None and not df.empty:
                # Convert DataFrame to list of dictionaries
                stock_records = []
                for _, row in df.iterrows():
                    record = {
                        'ticker': ticker,
                        'date': row['Datetime'] if 'Datetime' in row else row.name,
                        'open_price': float(row['Open']),
                        'high_price': float(row['High']),
                        'low_price': float(row['Low']),
                        'close_price': float(row['Close']),
                        'volume': int(row['Volume']),
                        'adj_close': float(row['Close'])  # yfinance already provides adjusted close
                    }
                    stock_records.append(record)
                
                all_stock_data[ticker] = stock_records
            else:
                all_stock_data[ticker] = []
        
        total_records = sum(len(records) for records in all_stock_data.values())
        logger.info(f"Collected {total_records} stock data records across {len(self.tickers)} tickers")
        
        return all_stock_data
    
    def collect_intraday_data(self, interval: str = "5m") -> Dict[str, List[Dict[str, Any]]]:
        """Collect intraday stock data for all configured tickers."""
        all_stock_data = {}
        
        for ticker in self.tickers:
            logger.info(f"Collecting intraday data for {ticker}")
            
            # Get intraday data for the current day
            df = self.get_stock_data(ticker, period="1d", interval=interval)
            
            if df is not None and not df.empty:
                # Convert DataFrame to list of dictionaries
                stock_records = []
                for _, row in df.iterrows():
                    record = {
                        'ticker': ticker,
                        'date': row['Datetime'] if 'Datetime' in row else row.name,
                        'open_price': float(row['Open']),
                        'high_price': float(row['High']),
                        'low_price': float(row['Low']),
                        'close_price': float(row['Close']),
                        'volume': int(row['Volume']),
                        'adj_close': float(row['Close'])
                    }
                    stock_records.append(record)
                
                all_stock_data[ticker] = stock_records
            else:
                all_stock_data[ticker] = []
        
        total_records = sum(len(records) for records in all_stock_data.values())
        logger.info(f"Collected {total_records} intraday records across {len(self.tickers)} tickers")
        
        return all_stock_data
    
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get detailed stock information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            stock_info = {
                'ticker': ticker,
                'company_name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'updated_at': datetime.utcnow()
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            return None
    
    def calculate_volatility(self, ticker: str, days: int = 30) -> Optional[float]:
        """Calculate stock volatility over a specified period."""
        try:
            df = self.get_stock_data(ticker, period=f"{days}d", interval="1d")
            
            if df is None or df.empty:
                return None
            
            # Calculate daily returns
            df['daily_return'] = df['Close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            volatility = df['daily_return'].std()
            
            # Annualize volatility
            annualized_volatility = volatility * (252 ** 0.5)  # 252 trading days in a year
            
            return float(annualized_volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {ticker}: {e}")
            return None
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get overall market summary for all tracked tickers."""
        summary = {
            'timestamp': datetime.utcnow(),
            'tickers_analyzed': len(self.tickers),
            'market_data': {}
        }
        
        for ticker in self.tickers:
            current_data = self.get_current_price(ticker)
            if current_data:
                # Calculate price change percentage
                if current_data['previous_close'] and current_data['current_price']:
                    price_change = current_data['current_price'] - current_data['previous_close']
                    price_change_pct = (price_change / current_data['previous_close']) * 100
                else:
                    price_change_pct = 0
                
                summary['market_data'][ticker] = {
                    'current_price': current_data['current_price'],
                    'price_change_pct': price_change_pct,
                    'volume': current_data['volume'],
                    'day_high': current_data['day_high'],
                    'day_low': current_data['day_low']
                }
        
        return summary