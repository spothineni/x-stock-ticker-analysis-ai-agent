"""Stock price prediction module using sentiment and technical analysis."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
from .analyzer import StockAnalyzer
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class StockPredictor:
    """Predicts stock price ranges using sentiment and technical analysis."""
    
    def __init__(self):
        """Initialize stock predictor."""
        self.analyzer = StockAnalyzer()
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'avg_sentiment', 'sentiment_std', 'tweet_count', 'avg_confidence',
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'volatility',
            'price_change', 'sma_5', 'sma_10', 'ema_5', 'ema_10'
        ]
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info("Stock predictor initialized")
    
    def prepare_features(self, combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning models."""
        try:
            df = combined_df.copy()
            
            # Calculate additional features
            df['bb_position'] = self._calculate_bb_position_series(df)
            
            # Create target variable (next day's price change)
            df['target'] = df['close_price'].shift(-1) / df['close_price'] - 1
            
            # Remove rows with missing target (last row)
            df = df[:-1]
            
            # Select feature columns that exist in the dataframe
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            if not available_features:
                logger.error("No feature columns available")
                return None, None
            
            # Fill missing values
            for col in available_features:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean() if df[col].dtype in ['float64', 'int64'] else 0)
            
            X = df[available_features]
            y = df['target']
            
            # Remove rows with missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None
    
    def _calculate_bb_position_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Band position for entire series."""
        try:
            bb_upper = df.get('bb_upper', df['close_price'])
            bb_lower = df.get('bb_lower', df['close_price'])
            price = df['close_price']
            
            # Avoid division by zero
            band_width = bb_upper - bb_lower
            band_width = band_width.replace(0, 1)  # Replace 0 with 1 to avoid division by zero
            
            position = (price - bb_lower) / band_width
            return position.clip(0, 1)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating BB position series: {e}")
            return pd.Series([0.5] * len(df))
    
    def train_model(self, ticker: str, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train prediction model for a specific ticker."""
        try:
            logger.info(f"Training {model_type} model for {ticker}")
            
            # Get combined data
            combined_df = self.analyzer.combine_stock_and_sentiment_data(ticker, days_back=60)
            
            if combined_df is None or len(combined_df) < 20:
                logger.warning(f"Insufficient data for training model for {ticker}")
                return {
                    'success': False,
                    'error': 'Insufficient data',
                    'ticker': ticker,
                    'model_type': model_type
                }
            
            # Prepare features
            X, y = self.prepare_features(combined_df)
            
            if X is None or len(X) < 10:
                logger.warning(f"Insufficient features for training model for {ticker}")
                return {
                    'success': False,
                    'error': 'Insufficient features',
                    'ticker': ticker,
                    'model_type': model_type
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'linear_regression':
                model = LinearRegression()
            else:
                logger.error(f"Unknown model type: {model_type}")
                return {
                    'success': False,
                    'error': f'Unknown model type: {model_type}',
                    'ticker': ticker,
                    'model_type': model_type
                }
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Store model and scaler
            model_key = f"{ticker}_{model_type}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Save model to disk
            model_path = self.model_dir / f"{model_key}_model.joblib"
            scaler_path = self.model_dir / f"{model_key}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Feature importance (for tree-based models)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            result = {
                'success': True,
                'ticker': ticker,
                'model_type': model_type,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                },
                'feature_importance': feature_importance,
                'trained_at': datetime.utcnow()
            }
            
            logger.info(f"Model training completed for {ticker}. Test R²: {test_r2:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'model_type': model_type
            }
    
    def load_model(self, ticker: str, model_type: str = 'random_forest') -> bool:
        """Load trained model from disk."""
        try:
            model_key = f"{ticker}_{model_type}"
            model_path = self.model_dir / f"{model_key}_model.joblib"
            scaler_path = self.model_dir / f"{model_key}_scaler.joblib"
            
            if not model_path.exists() or not scaler_path.exists():
                return False
            
            self.models[model_key] = joblib.load(model_path)
            self.scalers[model_key] = joblib.load(scaler_path)
            
            logger.info(f"Loaded model for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {e}")
            return False
    
    def predict_price_range(self, ticker: str, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Predict price range for next trading day."""
        try:
            model_key = f"{ticker}_{model_type}"
            
            # Try to load model if not in memory
            if model_key not in self.models:
                if not self.load_model(ticker, model_type):
                    # Train model if not available
                    training_result = self.train_model(ticker, model_type)
                    if not training_result['success']:
                        return {
                            'success': False,
                            'error': 'Could not train or load model',
                            'ticker': ticker
                        }
            
            # Get latest data
            combined_df = self.analyzer.combine_stock_and_sentiment_data(ticker, days_back=30)
            
            if combined_df is None or len(combined_df) == 0:
                return {
                    'success': False,
                    'error': 'No data available for prediction',
                    'ticker': ticker
                }
            
            # Prepare features for the latest data point
            latest_data = combined_df.iloc[-1:].copy()
            latest_data['bb_position'] = self._calculate_bb_position_series(latest_data)
            
            # Select available features
            available_features = [col for col in self.feature_columns if col in latest_data.columns]
            
            if not available_features:
                return {
                    'success': False,
                    'error': 'No features available for prediction',
                    'ticker': ticker
                }
            
            # Fill missing values
            for col in available_features:
                if col in latest_data.columns:
                    latest_data[col] = latest_data[col].fillna(0)
            
            X_latest = latest_data[available_features]
            
            # Scale features
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            X_scaled = scaler.transform(X_latest)
            
            # Make prediction
            predicted_change = model.predict(X_scaled)[0]
            
            # Get current price
            current_price = float(combined_df.iloc[-1]['close_price'])
            
            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_change)
            
            # Estimate confidence interval (using model's prediction uncertainty)
            # For ensemble models, we can use the standard deviation of individual tree predictions
            if hasattr(model, 'estimators_'):
                # Random Forest - get predictions from individual trees
                tree_predictions = np.array([tree.predict(X_scaled)[0] for tree in model.estimators_])
                prediction_std = np.std(tree_predictions)
            else:
                # For linear models, use a simple heuristic based on historical volatility
                volatility = combined_df['volatility'].iloc[-10:].mean()  # Last 10 days average volatility
                prediction_std = volatility * 0.5  # Rough estimate
            
            # Calculate prediction range (confidence interval)
            confidence_level = 0.68  # ~1 standard deviation
            range_multiplier = 1.0  # Adjust based on confidence level
            
            predicted_low = current_price * (1 + predicted_change - prediction_std * range_multiplier)
            predicted_high = current_price * (1 + predicted_change + prediction_std * range_multiplier)
            
            # Calculate prediction confidence based on model performance and data quality
            model_performance = getattr(model, 'score', lambda x, y: 0.5)(X_scaled, [predicted_change])
            data_quality = min(1.0, len(combined_df) / 30.0)  # More data = higher confidence
            sentiment_confidence = float(combined_df.iloc[-1]['avg_confidence'])
            
            overall_confidence = (model_performance * 0.4 + data_quality * 0.3 + sentiment_confidence * 0.3)
            overall_confidence = max(0.1, min(0.9, overall_confidence))  # Clamp between 0.1 and 0.9
            
            result = {
                'success': True,
                'ticker': ticker,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change_percent': predicted_change * 100,
                'predicted_range_low': predicted_low,
                'predicted_range_high': predicted_high,
                'prediction_confidence': overall_confidence,
                'model_type': model_type,
                'prediction_date': datetime.utcnow(),
                'features_used': available_features,
                'sentiment_score': float(combined_df.iloc[-1]['avg_sentiment']),
                'tweet_count': int(combined_df.iloc[-1]['tweet_count'])
            }
            
            logger.info(f"Price prediction completed for {ticker}: {predicted_change*100:.2f}% change")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting price range for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    def batch_predict_all_tickers(self, model_type: str = 'random_forest') -> Dict[str, Dict[str, Any]]:
        """Generate predictions for all configured tickers."""
        predictions = {}
        
        for ticker in config.stock_tickers:
            logger.info(f"Generating prediction for {ticker}")
            prediction = self.predict_price_range(ticker, model_type)
            predictions[ticker] = prediction
        
        successful_predictions = sum(1 for p in predictions.values() if p.get('success', False))
        logger.info(f"Generated {successful_predictions}/{len(config.stock_tickers)} successful predictions")
        
        return predictions
    
    def evaluate_model_performance(self, ticker: str, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Evaluate model performance using historical data."""
        try:
            # Get historical data
            combined_df = self.analyzer.combine_stock_and_sentiment_data(ticker, days_back=90)
            
            if combined_df is None or len(combined_df) < 30:
                return {
                    'success': False,
                    'error': 'Insufficient data for evaluation',
                    'ticker': ticker
                }
            
            # Prepare features
            X, y = self.prepare_features(combined_df)
            
            if X is None or len(X) < 20:
                return {
                    'success': False,
                    'error': 'Insufficient features for evaluation',
                    'ticker': ticker
                }
            
            # Use the last 20% of data for evaluation
            split_point = int(len(X) * 0.8)
            X_train, X_eval = X[:split_point], X[split_point:]
            y_train, y_eval = y[:split_point], y[split_point:]
            
            # Train model on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_eval_scaled = scaler.transform(X_eval)
            
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            else:
                model = LinearRegression()
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_eval_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_eval, y_pred)
            mae = mean_absolute_error(y_eval, y_pred)
            r2 = r2_score(y_eval, y_pred)
            
            # Calculate directional accuracy (did we predict the right direction?)
            actual_directions = np.sign(y_eval)
            predicted_directions = np.sign(y_pred)
            directional_accuracy = np.mean(actual_directions == predicted_directions)
            
            return {
                'success': True,
                'ticker': ticker,
                'model_type': model_type,
                'evaluation_samples': len(X_eval),
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'directional_accuracy': directional_accuracy
                },
                'evaluated_at': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }