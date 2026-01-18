# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import numpy as np
import sys
from transaction_processor import TransactionProcessor
from insights_generator import InsightsGenerator
from dotenv import load_dotenv

class PortfolioDashboard:
    """Streamlit dashboard for portfolio insights."""
    
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(
            page_title="Portfolio Insights Generator",
            page_icon="📊",
            layout="wide"
        )
        
        load_dotenv()
        
        # Initialize session state
        if 'processor' not in st.session_state:
            st.session_state.processor = None
        if 'analytics' not in st.session_state:
            st.session_state.analytics = None
        if 'insights' not in st.session_state:
            st.session_state.insights = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = None
        if 'ai_provider' not in st.session_state:
            st.session_state.ai_provider = "Gemini"  # Default to Gemini
        if 'custom_prompt' not in st.session_state:
            st.session_state.custom_prompt = ""  # Initialize custom_prompt
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.title("📊 Portfolio Insights Generator")
        st.markdown("---")
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Data Configuration")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload transaction CSV",
                type=['csv'],
                help="Upload a CSV file with transaction data"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("temp_transactions.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and process data
                if st.button("Load & Analyze Data") or not st.session_state.data_loaded:
                    with st.spinner("Processing transaction data..."):
                        try:
                            self.processor = TransactionProcessor("temp_transactions.csv")
                            self.processor.clean_data()
                            st.session_state.analytics = self.processor.analyze_data()
                            st.session_state.processor = self.processor
                            st.session_state.data_loaded = True
                            st.success("Data loaded successfully!")
                            
                            # Show summary
                            stats = st.session_state.analytics.get('summary_stats', {})
                            st.info(f"""
                            **Summary:**
                            - {stats.get('total_transactions', 0):,} transactions
                            - {stats.get('unique_tickers', 0)} tickers
                            - {stats.get('unique_traders', 0)} traders
                            - ${stats.get('total_value', 0):,.2f} total value
                            """)
                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")
            
            st.markdown("---")
            st.header("AI Configuration")
            
            # Provider selection
            provider = st.selectbox(
                "Select AI Provider",
                ["Gemini", "OpenAI"],
                index=0,  # Default to Gemini
                help="Google Gemini has a generous free tier. OpenAI requires credits."
            )
            st.session_state.ai_provider = provider
            
            # API key input based on provider
            if provider == "Gemini":
                api_key_input = st.text_input(
                    "Google Gemini API Key",
                    type="password",
                    help="Get free API key from: https://makersuite.google.com/app/apikey",
                    value=st.session_state.get('gemini_api_key', '')
                )
                
                if api_key_input:
                    st.session_state.gemini_api_key = api_key_input
                    os.environ['GEMINI_API_KEY'] = api_key_input
                    st.success("Gemini API key saved!")
            else:
                api_key_input = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Get API key from: https://platform.openai.com/api-keys",
                    value=st.session_state.get('openai_api_key', '')
                )
                
                if api_key_input:
                    st.session_state.openai_api_key = api_key_input
                    os.environ['OPENAI_API_KEY'] = api_key_input
                    st.success("OpenAI API key saved!")
            
            # Test API button
            if st.button("Test API Connection"):
                api_key = None
                provider_name = st.session_state.ai_provider
                if provider_name == "Gemini":
                    api_key = st.session_state.get('gemini_api_key')
                else:
                    api_key = st.session_state.get('openai_api_key')
                    
                if api_key:
                    with st.spinner("Testing API connection..."):
                        try:
                            if provider_name == "Gemini":
                                # Try new SDK first
                                try:
                                    from google import genai  # type: ignore
                                    client = genai.Client(api_key=api_key)  # type: ignore
                                    # Try multiple model names
                                    model_names = ['gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-pro']
                                    response = None
                                    for model_name in model_names:
                                        try:
                                            response = client.models.generate_content(  # type: ignore
                                                model=model_name,
                                                contents="Say 'API test successful'"
                                            )
                                            break
                                        except Exception:
                                            continue
                                    if response is None:
                                        raise Exception("Could not find a valid Gemini model")
                                    st.success(f"✅ Gemini API is valid! Response: {response.text}")
                                except ImportError:
                                    raise Exception("google-genai package not found. Install with: pip install google-genai")
                            else:
                                import openai
                                client = openai.OpenAI(api_key=api_key)
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "user", "content": "Say 'API test successful'"}],
                                    max_tokens=10
                                )
                                st.success(f"✅ OpenAI API is valid! Response: {response.choices[0].message.content}")
                        except Exception as e:
                            error_msg = str(e)
                            if "quota" in error_msg or "429" in error_msg or "insufficient_quota" in error_msg:
                                st.error(f"❌ API quota exceeded. Try using Gemini instead (free tier available).")
                            else:
                                st.error(f"❌ API error: {error_msg}")
                else:
                    st.warning("Please enter an API key first")
            
            st.markdown("---")
            st.header("About")
            st.info(
                "This dashboard analyzes trading patterns and generates AI-powered insights. "
                "Upload your transaction data to get started."
            )
        
        # Main content area
        if st.session_state.data_loaded and st.session_state.processor:
            self.display_dashboard()
        else:
            self.display_welcome()
    
    def display_welcome(self):
        """Display welcome message and instructions."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to Portfolio Insights Generator! 🚀
            
            This tool helps you:
            
            1. **Process** financial transaction data
            2. **Analyze** trading patterns and metrics
            3. **Generate** AI-powered insights
            
            ### Getting Started:
            
            1. Upload your transaction CSV file using the sidebar
            2. Click "Load & Analyze Data"
            3. Explore the analytics dashboard
            4. Generate insights with AI
            
            ### Sample Data Format:
            
            Your CSV should contain these columns:
            - `timestamp`: When the transaction occurred
            - `ticker`: Stock symbol (AAPL, GOOGL, etc.)
            - `action`: BUY or SELL
            - `quantity`: Number of shares
            - `price`: Price per share
            - `trader_id`: Trader identifier
            
            ### Try it with the sample data:
            """)
            
            # Load sample data button
            if st.button("Load Sample Data", type="primary"):
                with st.spinner("Loading sample data..."):
                    try:
                        # Use the provided sample data
                        self.processor = TransactionProcessor("sample_transactions.csv")
                        self.processor.clean_data()
                        st.session_state.analytics = self.processor.analyze_data()
                        st.session_state.processor = self.processor
                        st.session_state.data_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
    
    def display_dashboard(self):
        """Display the main dashboard with analytics and insights."""
        processor = st.session_state.processor
        analytics = st.session_state.analytics
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview",
            "🔍 Detailed Analytics",
            "🤖 AI Insights",
            "📋 Raw Data"
        ])
        
        with tab1:
            self.display_overview(processor, analytics)
        
        with tab2:
            self.display_detailed_analytics(processor, analytics)
        
        with tab3:
            self.display_insights_tab(processor, analytics)
        
        with tab4:
            self.display_raw_data(processor)
    
    def display_overview(self, processor, analytics):
        """Display overview metrics and charts."""
        st.header("Dashboard Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{analytics['summary_stats']['total_transactions']:,}",
                help="Total number of transactions"
            )
        
        with col2:
            st.metric(
                "Total Volume",
                f"{analytics['summary_stats']['total_volume']:,.0f}",
                help="Total shares traded"
            )
        
        with col3:
            st.metric(
                "Total Value",
                f"${analytics['summary_stats']['total_value']:,.2f}",
                help="Total transaction value"
            )
        
        with col4:
            st.metric(
                "Unique Traders",
                analytics['summary_stats']['unique_traders'],
                help="Number of unique traders"
            )
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Top tickers by volume
            st.subheader("Top 10 Tickers by Volume")
            top_tickers = processor.get_top_tickers(10, 'volume')
            if not top_tickers.empty:
                fig = px.bar(
                    x=top_tickers.index,
                    y=top_tickers.values,
                    labels={'x': 'Ticker', 'y': 'Volume (shares)'},
                    color=top_tickers.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No ticker data available")
        
        with col2:
            # Net positions
            st.subheader("Top 10 Net Positions")
            net_positions = pd.Series(analytics['net_position'])
            if not net_positions.empty:
                net_positions = net_positions.sort_values(ascending=False).head(10)
                
                # Create positive/negative colors
                colors = ['green' if x > 0 else 'red' for x in net_positions.values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=net_positions.index,
                        y=net_positions.values,
                        marker_color=colors,
                        text=[f"{x:+,.0f}" for x in net_positions.values],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    yaxis_title="Net Position (Shares)",
                    xaxis_title="Ticker",
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No net position data available")
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Most active traders
            st.subheader("Most Active Traders (by count)")
            top_traders = processor.get_top_traders(10, 'count')
            
            if not top_traders.empty:
                fig = px.bar(
                    x=top_traders.index,
                    y=top_traders.values,
                    labels={'x': 'Trader ID', 'y': 'Transaction Count'},
                    color=top_traders.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No trader data available")
        
        with col2:
            # Trading by hour
            st.subheader("Trading Activity by Hour")
            hourly_data = analytics.get('hourly_activity', {})
            
            if hourly_data and 'transaction_count' in hourly_data:
                hour_counts = hourly_data['transaction_count']
                hours = []
                counts = []
                for hour_str in sorted(hour_counts.keys(), key=int):
                    hours.append(int(hour_str))
                    counts.append(hour_counts[hour_str])
                
                fig = px.line(
                    x=hours,
                    y=counts,
                    labels={'x': 'Hour of Day', 'y': 'Transaction Count'},
                    markers=True
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No hourly data available")
    
    def display_detailed_analytics(self, processor, analytics):
        """Display detailed analytics and metrics."""
        st.header("Detailed Analytics")
        
        # Ticker Analysis
        st.subheader("Ticker Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Get ticker list
            volume_data = analytics.get('volume_per_ticker', {})
            if volume_data:
                ticker_list = list(volume_data.keys())
                selected_ticker = st.selectbox(
                    "Select Ticker for Detailed View",
                    options=ticker_list,
                    key="ticker_select"
                )
            else:
                st.warning("No ticker data available")
                selected_ticker = None
        
        if selected_ticker:
            try:
                ticker_data = processor.get_transactions_by_ticker(selected_ticker)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_volume = volume_data.get(selected_ticker, 0)
                    st.metric(f"{selected_ticker} Total Volume", f"{total_volume:,.0f}")
                
                with col2:
                    net_pos = analytics.get('net_position', {}).get(selected_ticker, 0)
                    st.metric(f"{selected_ticker} Net Position", f"{net_pos:+,.0f}")
                
                with col3:
                    total_value = analytics.get('value_per_ticker', {}).get(selected_ticker, 0)
                    st.metric(f"{selected_ticker} Total Value", f"${total_value:,.2f}")
                
                with col4:
                    price_stats = analytics.get('price_stats', {}).get(selected_ticker, {})
                    if isinstance(price_stats, dict):
                        avg_price = price_stats.get('mean', 0)
                    else:
                        avg_price = 0
                    st.metric(f"{selected_ticker} Avg Price", f"${avg_price:,.2f}")
                
                # Ticker transaction history
                st.subheader(f"{selected_ticker} Transaction History (Last 20)")
                if not ticker_data.empty:
                    display_data = ticker_data[['timestamp', 'action', 'quantity', 'price', 'trader_id']].tail(20)
                    display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(display_data, width='stretch')
                else:
                    st.info(f"No transactions found for {selected_ticker}")
            except Exception as e:
                st.error(f"Error loading ticker data: {str(e)}")
        
        # Trader Analysis
        st.subheader("Trader Analysis")
        trader_data = analytics.get('trader_activity', {})
        
        if trader_data:
            trader_list = list(trader_data.keys())
            if trader_list:
                selected_trader = st.selectbox("Select Trader for Detailed View", options=trader_list, key="trader_select")
                
                if selected_trader:
                    try:
                        trader_stats = trader_data.get(selected_trader, {})
                        trader_transactions = processor.get_transactions_by_trader(selected_trader)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if isinstance(trader_stats, dict):
                                count = trader_stats.get('transaction_count', 0)
                            else:
                                count = trader_stats
                            st.metric("Transaction Count", count)
                        
                        with col2:
                            if isinstance(trader_stats, dict):
                                value = trader_stats.get('transaction_value', 0)
                            else:
                                value = 0
                            st.metric("Total Value", f"${value:,.2f}")
                        
                        with col3:
                            if isinstance(trader_stats, dict):
                                volume = trader_stats.get('quantity', 0)
                            else:
                                volume = 0
                            st.metric("Total Volume", f"{volume:,.0f}")
                        
                        # Trader's ticker distribution
                        if not trader_transactions.empty:
                            st.subheader(f"{selected_trader}'s Trading Distribution")
                            trader_ticker_dist = trader_transactions.groupby('ticker')['quantity'].sum().sort_values(ascending=False)
                            
                            if len(trader_ticker_dist) > 0:
                                fig = px.pie(
                                    values=trader_ticker_dist.values,
                                    names=trader_ticker_dist.index,
                                    title=f"{selected_trader}'s Volume by Ticker"
                                )
                                st.plotly_chart(fig, width='stretch')
                        else:
                            st.info(f"No transaction data found for trader {selected_trader}")
                    except Exception as e:
                        st.error(f"Error loading trader data: {str(e)}")
            else:
                st.info("No trader data available")
        else:
            st.info("No trader data available")
        
        # Export options
        st.subheader("Export Analytics")
        if st.button("Export Analytics Report"):
            try:
                processor.export_analytics()
                st.success("Analytics report exported to 'analytics_report.json'")
            except Exception as e:
                st.error(f"Error exporting analytics: {str(e)}")
    
    def display_insights_tab(self, processor, analytics):
        """Display AI insights generation interface."""
        st.header("AI-Powered Insights Generation")
        
        # Get provider and API key from session state
        provider = st.session_state.get('ai_provider', 'Gemini').lower()
        
        api_key = None
        if provider == 'gemini':
            api_key = st.session_state.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')
            model = "gemini-3-flash-preview"
        else:
            api_key = st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            model = "gpt-3.5-turbo"
        
        # Initialize insights generator
        insights_gen = None
        if api_key:
            try:
                insights_gen = InsightsGenerator(api_key=api_key, model=model, provider=provider)
                st.success(f"✅ {provider.upper()} API is configured!")
            except Exception as e:
                st.error(f"Error initializing InsightsGenerator: {str(e)}")
                insights_gen = None
        else:
            st.warning(f"⚠️ {provider.upper()} API key not found. Please add your API key in the sidebar.")
            st.info("You can still view analytics in other tabs.")
        
        # Insights generation controls
        st.subheader("Generate Insights")
        
        # Apply pending prompt update before widget creation
        if '_pending_prompt' in st.session_state:
            st.session_state.custom_prompt = st.session_state._pending_prompt
            del st.session_state._pending_prompt
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Use value parameter to ensure it reads from session state
            custom_prompt = st.text_area(
                "Custom Prompt (optional)",
                value=st.session_state.get('custom_prompt', ''),
                height=100,
                help="Customize the prompt for insights generation. Leave empty for default analysis.",
                key="custom_prompt"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            generate_disabled = insights_gen is None
            if st.button("Generate Insights", type="primary", width='stretch', disabled=generate_disabled):
                if insights_gen:
                    with st.spinner("Generating AI insights. This may take a few seconds..."):
                        try:
                            insights = insights_gen.generate_insights(analytics, custom_prompt)
                            st.session_state.insights = insights
                            st.success("Insights generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                else:
                    st.error("AI API is not configured. Please add your API key in the sidebar.")
        
        # Display generated insights
        if 'insights' in st.session_state and st.session_state.insights:
            insights = st.session_state.insights
            
            st.markdown("---")
            st.subheader("Generated Insights")
            
            # Raw response toggle
            with st.expander("View Raw AI Response"):
                raw_response = insights.get('raw_response', 'No raw response available')
                st.text_area("Raw Response", raw_response, height=300, key="raw_response")
            
            # Display structured sections
            sections = insights.get('sections', {})
            if sections:
                for section_name, section_content in sections.items():
                    with st.expander(f"📋 {section_name}"):
                        st.markdown(section_content)
            else:
                st.info("No structured sections found in the insights.")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Insights to File"):
                    try:
                        if insights_gen:
                            insights_gen.save_insights(insights)
                        else:
                            # Save manually
                            with open('generated_insights.json', 'w') as f:
                                json.dump(insights, f, indent=2, default=str)
                        st.success("Insights saved to 'generated_insights.json'")
                    except Exception as e:
                        st.error(f"Error saving insights: {str(e)}")
            
            with col2:
                if st.button("Generate New Insights"):
                    st.session_state.insights = None
                    st.rerun()
        
        # Prompt suggestions
        if not st.session_state.get('insights'):
            st.markdown("---")
            st.subheader("Prompt Suggestions:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Analyze Risk Patterns", disabled=generate_disabled, key="btn_risk"):
                    # Use a temporary key to avoid conflict, then update on rerun
                    st.session_state._pending_prompt = "Focus specifically on identifying concentration risks, unusual trading patterns, and potential compliance issues."
                    st.rerun()
            
            with col2:
                if st.button("Evaluate Trading Strategy", disabled=generate_disabled, key="btn_strategy"):
                    st.session_state._pending_prompt = "Analyze the trading patterns to identify what strategies traders might be using and evaluate their effectiveness."
                    st.rerun()
            
            with col3:
                if st.button("Market Sentiment Analysis", disabled=generate_disabled, key="btn_sentiment"):
                    st.session_state._pending_prompt = "Based on the trading data, what does this suggest about market sentiment and sector preferences?"
                    st.rerun()
    
    def display_raw_data(self, processor):
        """Display raw transaction data."""
        st.header("Raw Transaction Data")
        
        try:
            data = processor.processed_data
            
            # Show data preview
            st.subheader("Data Preview (First 100 rows)")
            preview_data = data.head(100).copy()
            preview_data['timestamp'] = preview_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(preview_data, use_container_width=True)
            
            # Data statistics
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Information:**")
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Non-Null Count': data.notnull().sum().values,
                    'Data Type': [str(dtype) for dtype in data.dtypes.values]
                })
                st.dataframe(col_info, use_container_width=True)
            
            with col2:
                st.write("**Summary Statistics:**")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_data = data[numeric_cols].describe()
                    st.dataframe(stats_data, use_container_width=True)
                else:
                    st.info("No numeric columns found")
            
            # Data export
            st.subheader("Export Data")
            if st.button("Export Processed Data to CSV"):
                data.to_csv('processed_transactions.csv', index=False)
                st.success("Data exported to 'processed_transactions.csv'")
        except Exception as e:
            st.error(f"Error displaying raw data: {str(e)}")

def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = PortfolioDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Error running dashboard: {str(e)}")
        st.write("Please check the data file and try again.")

if __name__ == "__main__":
    main()