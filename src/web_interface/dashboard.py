"""Dashboard routes for the web interface."""

from flask import Blueprint, render_template_string, jsonify
from datetime import datetime
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Sentiment Analyzer Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.9;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .status-bar {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-item {
            text-align: center;
        }
        
        .status-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .status-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.25rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .card h3 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        
        .ticker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
        }
        
        .ticker-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .ticker-symbol {
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .ticker-info {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.25rem;
            font-size: 0.9rem;
        }
        
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #95a5a6; }
        
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            margin: 0.25rem;
            transition: background-color 0.3s;
        }
        
        .btn:hover {
            background: #5a6fd8;
        }
        
        .btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .error {
            background: #e74c3c;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .success {
            background: #27ae60;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .actions {
            text-align: center;
            margin: 2rem 0;
        }
        
        .last-updated {
            text-align: center;
            color: #666;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 Stock Sentiment Analyzer</h1>
        <p>Real-time sentiment analysis and stock predictions based on Twitter/X data</p>
    </div>
    
    <div class="container">
        <div class="status-bar" id="statusBar">
            <div class="status-item">
                <div class="status-value" id="systemStatus">Loading...</div>
                <div class="status-label">System Status</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="tickerCount">-</div>
                <div class="status-label">Tracked Tickers</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="lastForecast">-</div>
                <div class="status-label">Last Forecast</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="tweetsAnalyzed">-</div>
                <div class="status-label">Tweets Analyzed</div>
            </div>
        </div>
        
        <div class="actions">
            <button class="btn" onclick="runForecast()">🔄 Run Forecast</button>
            <button class="btn" onclick="generateReport()">📊 Generate Report</button>
            <button class="btn" onclick="collectData()">📥 Collect Data</button>
            <button class="btn" onclick="refreshDashboard()">🔃 Refresh</button>
        </div>
        
        <div id="messages"></div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Market Overview</h3>
                <div id="marketOverview">
                    <div class="loading">Loading market data...</div>
                </div>
            </div>
            
            <div class="card">
                <h3>💭 Sentiment Distribution</h3>
                <div class="chart-container">
                    <canvas id="sentimentChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Stock Tickers Analysis</h3>
            <div class="ticker-grid" id="tickerGrid">
                <div class="loading">Loading ticker data...</div>
            </div>
        </div>
        
        <div class="card">
            <h3>🔮 Latest Predictions</h3>
            <div id="predictions">
                <div class="loading">Loading predictions...</div>
            </div>
        </div>
        
        <div class="last-updated" id="lastUpdated">
            Last updated: <span id="updateTime">-</span>
        </div>
    </div>

    <script>
        let sentimentChart = null;
        
        // Initialize dashboard
        $(document).ready(function() {
            refreshDashboard();
            
            // Auto-refresh every 5 minutes
            setInterval(refreshDashboard, 5 * 60 * 1000);
        });
        
        function showMessage(message, type = 'success') {
            const messagesDiv = $('#messages');
            const messageHtml = `<div class="${type}">${message}</div>`;
            messagesDiv.html(messageHtml);
            
            setTimeout(() => {
                messagesDiv.empty();
            }, 5000);
        }
        
        function refreshDashboard() {
            console.log('refreshDashboard() called');
            loadSystemStatus();
            console.log('loadSystemStatus() called');
            loadMarketOverview();
            console.log('loadMarketOverview() called');
            loadTickerData();
            console.log('loadTickerData() called');
            loadPredictions();
            console.log('loadPredictions() called');
            updateTimestamp();
            console.log('updateTimestamp() called');
        }
        
        function loadSystemStatus() {
            $.get('/api/status')
                .done(function(data) {
                    $('#systemStatus').text('🟢 Online');
                    $('#tickerCount').text(data.configured_tickers.length);
                })
                .fail(function() {
                    $('#systemStatus').text('🔴 Offline');
                });
            
            // Load forecast summary
            $.get('/api/forecast-summary')
                .done(function(data) {
                    if (data.success) {
                        const date = new Date(data.forecast_date).toLocaleDateString();
                        $('#lastForecast').text(date);
                        
                        if (data.summary && data.summary.tweets_analyzed) {
                            $('#tweetsAnalyzed').text(data.summary.tweets_analyzed);
                        }
                    }
                });
        }
        
        function loadMarketOverview() {
            console.log('loadMarketOverview() function started');
            $('#marketOverview').html('<div>Loading market data...</div>');
            
            $.get('/api/market-summary')
                .done(function(data) {
                    console.log('loadMarketOverview() API call successful:', data);
                    $('#marketOverview').html('<div>API call successful! Processing data...</div>');
                    
                    setTimeout(function() {
                        try {
                            if (data.success) {
                                const summary = data.market_summary;
                                const sentiment = summary.sentiment_overview;
                                
                                let html = '<div>';
                                html += `<div class="ticker-info"><span>Tickers Analyzed:</span><span>${summary.tickers_analyzed}</span></div>`;
                                html += `<div class="ticker-info"><span>Average Sentiment:</span><span class="${getSentimentClass(sentiment.avg_sentiment)}">${sentiment.avg_sentiment.toFixed(3)}</span></div>`;
                                html += `<div class="ticker-info"><span>Total Tweets:</span><span>${sentiment.total_tweets_analyzed}</span></div>`;
                                html += '</div>';
                                
                                $('#marketOverview').html(html);
                                console.log('Market overview HTML updated');
                                
                                // Update sentiment chart
                                console.log('About to update sentiment chart with:', sentiment.sentiment_distribution);
                                updateSentimentChart(sentiment.sentiment_distribution);
                                console.log('Sentiment chart updated');
                            } else {
                                console.error('API returned success=false');
                                $('#marketOverview').html('<div class="error">API returned error</div>');
                            }
                        } catch (error) {
                            console.error('Error processing market overview data:', error);
                            $('#marketOverview').html('<div class="error">Error processing data: ' + error.message + '</div>');
                        }
                    }, 100);
                })
                .fail(function(xhr, status, error) {
                    console.error('loadMarketOverview() API call failed:', error);
                    $('#marketOverview').html('<div class="error">Failed to load market data</div>');
                });
        }
        
        function loadTickerData() {
            console.log('loadTickerData() function started');
            $('#tickerGrid').html('<div>Loading ticker data...</div>');
            
            $.get('/api/tickers')
                .done(function(data) {
                    console.log('loadTickerData() API call successful:', data);
                    $('#tickerGrid').html('<div>API call successful! Found ' + data.tickers.length + ' tickers. Loading analysis...</div>');
                    
                    const tickers = data.tickers;
                    
                    setTimeout(function() {
                        // Clear the loading message
                        $('#tickerGrid').empty();
                        
                        tickers.forEach(ticker => {
                            loadTickerAnalysis(ticker);
                        });
                    }, 100);
                })
                .fail(function(xhr, status, error) {
                    console.error('loadTickerData() API call failed:', error);
                    $('#tickerGrid').html('<div class="error">Failed to load ticker data</div>');
                });
        }
        
        function loadTickerAnalysis(ticker) {
            console.log(`loadTickerAnalysis() called for ${ticker}`);
            $.get(`/api/ticker/${ticker}/analysis`)
                .done(function(data) {
                    console.log(`API call successful for ${ticker}:`, data);
                    if (data.success) {
                        const analysis = data.analysis;
                        console.log(`Creating ticker card for ${ticker}`);
                        const html = createTickerCard(ticker, analysis);
                        console.log(`Ticker card HTML created for ${ticker}:`, html.substring(0, 100) + '...');
                        
                        // Update or add ticker card
                        const existingCard = $(`#ticker-${ticker}`);
                        if (existingCard.length) {
                            console.log(`Replacing existing card for ${ticker}`);
                            existingCard.replaceWith(html);
                        } else {
                            console.log(`Appending new card for ${ticker}`);
                            $('#tickerGrid').append(html);
                        }
                        console.log(`Ticker card added for ${ticker}`);
                    } else {
                        console.error(`Failed to load analysis for ${ticker}:`, data);
                        $('#tickerGrid').append(`<div>Error loading ${ticker}: API returned success=false</div>`);
                    }
                })
                .fail(function(xhr, status, error) {
                    console.error(`Error loading analysis for ${ticker}:`, error);
                    $('#tickerGrid').append(`<div>Error loading ${ticker}: ${error}</div>`);
                });
        }
        
        function createTickerCard(ticker, analysis) {
            console.log(`Creating ticker card for ${ticker}:`, analysis);
            
            const sentimentClass = getSentimentClass(analysis.avg_sentiment);
            const priceClass = analysis.price_change_percent > 0 ? 'positive' : 
                              analysis.price_change_percent < 0 ? 'negative' : 'neutral';
            
            // Safe formatting with fallbacks for NaN/null values
            const avgSentiment = (analysis.avg_sentiment != null && !isNaN(analysis.avg_sentiment)) ? 
                                analysis.avg_sentiment.toFixed(3) : '0.000';
            const priceChange = (analysis.price_change_percent != null && !isNaN(analysis.price_change_percent)) ? 
                               analysis.price_change_percent.toFixed(2) : '0.00';
            const volumeChange = (analysis.volume_change_percent != null && !isNaN(analysis.volume_change_percent)) ? 
                                analysis.volume_change_percent.toFixed(1) : '0.0';
            
            return `
                <div class="ticker-card" id="ticker-${ticker}">
                    <div class="ticker-symbol">${ticker}</div>
                    <div class="ticker-info">
                        <span>Sentiment:</span>
                        <span class="${sentimentClass}">${avgSentiment}</span>
                    </div>
                    <div class="ticker-info">
                        <span>Price Change:</span>
                        <span class="${priceClass}">${priceChange}%</span>
                    </div>
                    <div class="ticker-info">
                        <span>Tweets:</span>
                        <span>${analysis.tweet_count || 0}</span>
                    </div>
                    <div class="ticker-info">
                        <span>Volume Change:</span>
                        <span>${volumeChange}%</span>
                    </div>
                </div>
            `;
        }
        
        function loadPredictions() {
            $.get('/api/tickers')
                .done(function(data) {
                    const tickers = data.tickers;
                    let html = '<div class="ticker-grid">';
                    
                    let loadedCount = 0;
                    tickers.forEach(ticker => {
                        $.get(`/api/ticker/${ticker}/prediction`)
                            .done(function(predData) {
                                if (predData.success) {
                                    const pred = predData;
                                    const changeClass = pred.predicted_change_percent > 0 ? 'positive' : 
                                                       pred.predicted_change_percent < 0 ? 'negative' : 'neutral';
                                    
                                    html += `
                                        <div class="ticker-card">
                                            <div class="ticker-symbol">${ticker}</div>
                                            <div class="ticker-info">
                                                <span>Current:</span>
                                                <span>$${pred.current_price.toFixed(2)}</span>
                                            </div>
                                            <div class="ticker-info">
                                                <span>Predicted:</span>
                                                <span class="${changeClass}">$${pred.predicted_price.toFixed(2)}</span>
                                            </div>
                                            <div class="ticker-info">
                                                <span>Change:</span>
                                                <span class="${changeClass}">${pred.predicted_change_percent.toFixed(2)}%</span>
                                            </div>
                                            <div class="ticker-info">
                                                <span>Confidence:</span>
                                                <span>${(pred.prediction_confidence * 100).toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    `;
                                }
                                
                                loadedCount++;
                                if (loadedCount === tickers.length) {
                                    html += '</div>';
                                    $('#predictions').html(html);
                                }
                            });
                    });
                });
        }
        
        function updateSentimentChart(distribution) {
            try {
                console.log('updateSentimentChart() called with:', distribution);
                const ctx = document.getElementById('sentimentChart').getContext('2d');
                
                if (sentimentChart) {
                    sentimentChart.destroy();
                }
                
                // Handle case where all values are 0 - show a placeholder chart
                const hasData = distribution.positive > 0 || distribution.negative > 0 || distribution.neutral > 0;
                const chartData = hasData ? 
                    [distribution.positive, distribution.negative, distribution.neutral] :
                    [1, 1, 1]; // Equal placeholder values when no data
                
                sentimentChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Positive', 'Negative', 'Neutral'],
                        datasets: [{
                            data: chartData,
                            backgroundColor: hasData ? ['#27ae60', '#e74c3c', '#95a5a6'] : ['#ddd', '#ddd', '#ddd'],
                            borderWidth: 2,
                            borderColor: '#fff'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
                console.log('Sentiment chart created successfully');
            } catch (error) {
                console.error('Error creating sentiment chart:', error);
            }
        }
        
        function getSentimentClass(sentiment) {
            if (sentiment > 0.1) return 'positive';
            if (sentiment < -0.1) return 'negative';
            return 'neutral';
        }
        
        function runForecast() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '🔄 Running...';
            
            $.post('/api/run-forecast')
                .done(function(data) {
                    if (data.success) {
                        showMessage('Forecast completed successfully!', 'success');
                        refreshDashboard();
                    } else {
                        showMessage('Forecast failed: ' + data.error, 'error');
                    }
                })
                .fail(function() {
                    showMessage('Failed to run forecast', 'error');
                })
                .always(function() {
                    btn.disabled = false;
                    btn.textContent = '🔄 Run Forecast';
                });
        }
        
        function generateReport() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '📊 Generating...';
            
            $.post('/api/generate-report')
                .done(function(data) {
                    if (data.success) {
                        showMessage('Report generated successfully!', 'success');
                    } else {
                        showMessage('Report generation failed: ' + data.error, 'error');
                    }
                })
                .fail(function() {
                    showMessage('Failed to generate report', 'error');
                })
                .always(function() {
                    btn.disabled = false;
                    btn.textContent = '📊 Generate Report';
                });
        }
        
        function collectData() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '📥 Collecting...';
            
            $.post('/api/collect-data')
                .done(function(data) {
                    if (data.success) {
                        showMessage('Data collection completed!', 'success');
                        refreshDashboard();
                    } else {
                        showMessage('Data collection failed', 'error');
                    }
                })
                .fail(function() {
                    showMessage('Failed to collect data', 'error');
                })
                .always(function() {
                    btn.disabled = false;
                    btn.textContent = '📥 Collect Data';
                });
        }
        
        function updateTimestamp() {
            $('#updateTime').text(new Date().toLocaleString());
        }
    </script>
</body>
</html>
"""

@dashboard_bp.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_HTML)

@dashboard_bp.route('/health')
def health():
    """Health check for dashboard."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'dashboard'
    })