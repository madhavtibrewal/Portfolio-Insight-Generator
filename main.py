#!/usr/bin/env python3
"""
Portfolio Insights Generator - Main Application

This application processes financial transaction data,
analyzes trading patterns, and generates AI-powered insights.
"""

import sys
import argparse
from transaction_processor import TransactionProcessor
from insights_generator import InsightsGenerator

def run_cli():
    """Run the application in CLI mode."""
    parser = argparse.ArgumentParser(description='Portfolio Insights Generator')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to transaction CSV file')
    parser.add_argument('--analyze', '-a', action='store_true', help='Run analytics only')
    parser.add_argument('--insights', '-i', action='store_true', help='Generate insights')
    parser.add_argument('--export', '-e', type=str, help='Export analytics to file')
    parser.add_argument('--api-key', '-k', type=str, help='OpenAI API key for insights')
    
    args = parser.parse_args()
    
    if args.file:
        try:
            # Initialize processor
            print(f"Loading data from {args.file}...")
            processor = TransactionProcessor(args.file)
            processor.clean_data()
            analytics = processor.analyze_data()
            
            if args.analyze:
                # Display summary
                summary = analytics['summary_stats']
                print("\n" + "="*50)
                print("ANALYTICS SUMMARY")
                print("="*50)
                print(f"Total Transactions: {summary['total_transactions']:,}")
                print(f"Unique Tickers: {summary['unique_tickers']}")
                print(f"Unique Traders: {summary['unique_traders']}")
                print(f"Total Volume: {summary['total_volume']:,.0f} shares")
                print(f"Total Value: ${summary['total_value']:,.2f}")
                print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
                
                # Top tickers
                print("\nTop 5 Tickers by Volume:")
                top_tickers = processor.get_top_tickers(5, 'volume')
                for ticker, volume in top_tickers.items():
                    print(f"  {ticker}: {volume:,.0f} shares")
            
            if args.insights:
                # Generate insights
                print("\n" + "="*50)
                print("GENERATING AI INSIGHTS...")
                print("="*50)
                
                insights_gen = InsightsGenerator(api_key=args.api_key)
                insights = insights_gen.generate_insights(analytics)
                
                print("\nINSIGHTS SUMMARY:")
                print("-"*50)
                sections = insights.get('sections', {})
                for section, content in sections.items():
                    print(f"\n{section.upper()}:")
                    print("-"*30)
                    # Print first 500 characters of each section
                    preview = content[:500]
                    print(preview)
                    if len(content) > 500:
                        print("...")
                
                # Save insights
                insights_gen.save_insights(insights, 'cli_insights.json')
                print(f"\nFull insights saved to cli_insights.json")
            
            if args.export:
                processor.export_analytics(args.export)
                print(f"\nAnalytics exported to {args.export}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        print("Error: Please provide a CSV file using --file option")
        parser.print_help()
        sys.exit(1)

def main():
    """Main entry point."""
    print("\n" + "="*50)
    print("PORTFOLIO INSIGHTS GENERATOR")
    print("="*50)
    print("\nSelect mode:")
    print("1. CLI Mode (analytics & insights)")
    print("2. Dashboard Mode (interactive web app)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_cli()
        elif choice == '2':
            print("\nLaunching dashboard...")
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        elif choice == '3':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please run with --help for CLI options.")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()