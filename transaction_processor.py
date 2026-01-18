import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

class TransactionProcessor:
    """Processes and analyzes financial transaction data."""
    
    def __init__(self, file_path: str = None):
        """
        Initialize the TransactionProcessor.
        
        Args:
            file_path (str): Path to the CSV file containing transaction data
        """
        self.transactions = None
        self.processed_data = None
        self.analytics = {}
        
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load transaction data from CSV file.
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded transaction data
        """
        try:
            # Load CSV with error handling
            self.transactions = pd.read_csv(file_path)
            
            # Basic validation of required columns
            required_columns = ['timestamp', 'ticker', 'action', 'quantity', 'price', 'trader_id']
            missing_columns = [col for col in required_columns if col not in self.transactions.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Successfully loaded {len(self.transactions)} transactions")
            return self.transactions
            
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the transaction data.
        
        Returns:
            pd.DataFrame: Cleaned transaction data
        """
        if self.transactions is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Create a copy to avoid modifying original
        cleaned_data = self.transactions.copy()
        
        # Convert timestamp to datetime
        cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'], errors='coerce')
        
        # Fill missing trader_id with 'UNKNOWN'
        cleaned_data['trader_id'] = cleaned_data['trader_id'].fillna('UNKNOWN')
        
        # Ensure quantity and price are numeric
        cleaned_data['quantity'] = pd.to_numeric(cleaned_data['quantity'], errors='coerce')
        cleaned_data['price'] = pd.to_numeric(cleaned_data['price'], errors='coerce')
        
        # Validate action column
        valid_actions = ['BUY', 'SELL']
        invalid_actions = cleaned_data[~cleaned_data['action'].isin(valid_actions)]
        if len(invalid_actions) > 0:
            print(f"Found {len(invalid_actions)} rows with invalid actions. Setting to NaN.")
            cleaned_data['action'] = cleaned_data['action'].where(
                cleaned_data['action'].isin(valid_actions), np.nan
            )
        
        # Calculate transaction value
        cleaned_data['transaction_value'] = cleaned_data['quantity'] * cleaned_data['price']
        
        # Remove rows with critical missing values
        initial_count = len(cleaned_data)
        cleaned_data = cleaned_data.dropna(subset=['timestamp', 'ticker', 'action', 'quantity', 'price'])
        removed_count = initial_count - len(cleaned_data)
        
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing critical data")
        
        # Remove duplicate rows
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Sort by timestamp
        cleaned_data = cleaned_data.sort_values('timestamp').reset_index(drop=True)
        
        self.processed_data = cleaned_data
        print(f"Data cleaning complete. {len(cleaned_data)} valid transactions remaining.")
        
        return cleaned_data
    
    def analyze_data(self) -> Dict:
        """
        Perform analytics on the transaction data.
        
        Returns:
            Dict: Dictionary containing all analytics results
        """
        if self.processed_data is None:
            self.clean_data()
        
        data = self.processed_data
        
        # 1. Total volume traded per ticker
        volume_per_ticker = data.groupby('ticker')['quantity'].sum().sort_values(ascending=False)
        
        # 2. Net position per ticker (total buys - total sells)
        buy_data = data[data['action'] == 'BUY'].groupby('ticker')['quantity'].sum()
        sell_data = data[data['action'] == 'SELL'].groupby('ticker')['quantity'].sum()
        net_position = (buy_data.fillna(0) - sell_data.fillna(0)).sort_values(ascending=False)
        
        # 3. Most active traders (by transaction count)
        trader_activity = data.groupby('trader_id').agg({
            'timestamp': 'count',
            'transaction_value': 'sum',
            'quantity': 'sum'
        }).rename(columns={'timestamp': 'transaction_count'}).sort_values('transaction_count', ascending=False)
        
        # 4. Most active traders by volume
        trader_volume = data.groupby('trader_id')['quantity'].sum().sort_values(ascending=False)
        
        # 5. Transaction value per ticker
        value_per_ticker = data.groupby('ticker')['transaction_value'].sum().sort_values(ascending=False)
        
        # 6. Time-based analysis
        data['hour'] = data['timestamp'].dt.hour
        data['date'] = data['timestamp'].dt.date
        data['day_of_week'] = data['timestamp'].dt.day_name()
        
        # Hourly activity - use default to_dict() to get column-oriented structure
        hourly_activity = data.groupby('hour').agg({
            'timestamp': 'count',
            'transaction_value': 'sum'
        }).rename(columns={'timestamp': 'transaction_count'}).to_dict()
        
        # Daily activity - use default to_dict() to get column-oriented structure
        daily_activity = data.groupby('date').agg({
            'timestamp': 'count',
            'transaction_value': 'sum',
            'quantity': 'sum'
        }).rename(columns={'timestamp': 'transaction_count'}).to_dict()
        
        # 7. Average transaction size per trader
        avg_txn_size = data.groupby('trader_id').agg({
            'quantity': 'mean',
            'transaction_value': 'mean'
        }).rename(columns={'quantity': 'avg_quantity', 'transaction_value': 'avg_value'})
        
        # 8. Price analysis per ticker
        price_stats = data.groupby('ticker')['price'].agg(['min', 'max', 'mean', 'std']).round(2).to_dict()
        
        # Store all analytics
        self.analytics = {
            'volume_per_ticker': volume_per_ticker.to_dict(),
            'net_position': net_position.to_dict(),
            'trader_activity': trader_activity.to_dict('index'),  # Use 'index' to get trader_id as keys
            'trader_volume': trader_volume.to_dict(),
            'value_per_ticker': value_per_ticker.to_dict(),
            'hourly_activity': hourly_activity,
            'daily_activity': daily_activity,
            'avg_txn_size': avg_txn_size.to_dict('index'),  # Use 'index' to get trader_id as keys
            'price_stats': price_stats,
            'summary_stats': {
                'total_transactions': len(data),
                'unique_tickers': data['ticker'].nunique(),
                'unique_traders': data['trader_id'].nunique(),
                'total_volume': data['quantity'].sum(),
                'total_value': data['transaction_value'].sum(),
                'date_range': {
                    'start': data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        }
        
        return self.analytics
    
    def get_transactions_by_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Get all transactions for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: Transactions for the specified ticker
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        return self.processed_data[self.processed_data['ticker'] == ticker].copy()
    
    def get_transactions_by_trader(self, trader_id: str) -> pd.DataFrame:
        """
        Get all transactions for a specific trader.
        
        Args:
            trader_id (str): Trader identifier
            
        Returns:
            pd.DataFrame: Transactions for the specified trader
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        return self.processed_data[self.processed_data['trader_id'] == trader_id].copy()
    
    def get_transactions_in_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get transactions within a specific time range.
        
        Args:
            start_time (datetime): Start of time range
            end_time (datetime): End of time range
            
        Returns:
            pd.DataFrame: Transactions within the specified time range
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        mask = (self.processed_data['timestamp'] >= start_time) & (self.processed_data['timestamp'] <= end_time)
        return self.processed_data[mask].copy()
    
    def get_top_tickers(self, n: int = 10, by: str = 'volume') -> pd.DataFrame:
        """
        Get top N tickers by specified metric.
        
        Args:
            n (int): Number of top tickers to return
            by (str): Metric to sort by ('volume', 'value', 'transactions')
            
        Returns:
            pd.DataFrame: Top N tickers
        """
        if not self.analytics:
            self.analyze_data()
        
        if by == 'volume':
            data = pd.Series(self.analytics['volume_per_ticker']).sort_values(ascending=False)
        elif by == 'value':
            data = pd.Series(self.analytics['value_per_ticker']).sort_values(ascending=False)
        else:
            raise ValueError(f"Invalid sort metric: {by}")
        
        return data.head(n)
    
    def get_top_traders(self, n: int = 10, by: str = 'count') -> pd.DataFrame:
        """
        Get top N traders by specified metric.
        
        Args:
            n (int): Number of top traders to return
            by (str): Metric to sort by ('count', 'volume', 'value')
            
        Returns:
            pd.DataFrame: Top N traders
        """
        if not self.analytics:
            self.analyze_data()
        
        trader_data = self.analytics['trader_activity']
        
        if by == 'count':
            # Create a Series from the nested dictionary
            counts = {}
            for trader_id, stats in trader_data.items():
                if isinstance(stats, dict):
                    counts[trader_id] = stats.get('transaction_count', 0)
                else:
                    counts[trader_id] = stats
            data = pd.Series(counts).sort_values(ascending=False)
        elif by == 'volume':
            data = pd.Series(self.analytics['trader_volume']).sort_values(ascending=False)
        elif by == 'value':
            # Create a Series from the nested dictionary
            values = {}
            for trader_id, stats in trader_data.items():
                if isinstance(stats, dict):
                    values[trader_id] = stats.get('transaction_value', 0)
                else:
                    values[trader_id] = stats
            data = pd.Series(values).sort_values(ascending=False)
        else:
            raise ValueError(f"Invalid sort metric: {by}")
        
        return data.head(n)
    
    def export_analytics(self, file_path: str = 'analytics_report.json'):
        """
        Export analytics results to a JSON file.
        
        Args:
            file_path (str): Path to save the JSON file
        """
        if not self.analytics:
            self.analyze_data()
        
        with open(file_path, 'w') as f:
            json.dump(self.analytics, f, indent=2, default=str)
        
        print(f"Analytics exported to {file_path}")
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the transaction data.
        
        Returns:
            Dict: Summary statistics
        """
        if not self.analytics:
            self.analyze_data()
        
        return self.analytics['summary_stats']