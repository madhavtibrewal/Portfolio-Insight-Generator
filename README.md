# URL
https://portfolio-insight-generator-z5xkvpgrvrxlvdgef3zxbw.streamlit.app/

# Portfolio Insights Generator

Built a Python app that processes transaction data, analyzes trading patterns, and generates AI insights. Simple pipeline: load CSV → clean → analyze → generate insights with LLM.

## What It Does

Three components:
1. **Transaction Processor** - Reads CSV, cleans messy data (missing values, duplicates), analyzes patterns
2. **Insights Generator** - Uses Google Gemini API to generate natural language insights
3. **Dashboard** - Streamlit web app for exploring data and generating insights

## Features

✅ **Data Processing**
- Load CSV transaction data
- Auto-clean: handle missing trader_id (fill with UNKNOWN), remove duplicates, validate data
- Query by ticker, trader, or time range
- Handles 1000+ transactions

✅ **Analytics**
- Volume per ticker
- Net position per ticker (buys - sells)
- Most active traders
- Hourly/daily trading patterns
- Price stats and concentration analysis

✅ **AI Insights**
- Generate insights using Google Gemini (free tier)
- Custom prompts for focused analysis
- Identifies patterns, risks, unusual activity
- Fallback mode if API fails (doesn't crash)

✅ **Dashboard**
- Interactive charts with Plotly
- Multiple tabs (Overview, Analytics, AI Insights, Raw Data)
- Upload data and explore instantly
- Export results to CSV/JSON

## Quick Start

### Setup (First Time)

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get API Key

Go to: https://makersuite.google.com/app/apikey
- Click "Create API Key"
- Copy the key
- Create `.env` file with: `GEMINI_API_KEY=your_key_here`

### Run Dashboard (Recommended)

```bash
streamlit run dashboard.py
```

Open http://localhost:8501 → Upload CSV → Explore → Generate insights

### Or Use CLI

```bash
python main.py
```

Choose CLI mode to analyze without the web interface.

## Data Format

Your CSV needs these columns:

| Column | Example |
|--------|---------|
| timestamp | 2024-01-15 19:02:00 |
| ticker | AAPL, GOOGL, MSFT |
| action | BUY or SELL |
| quantity | 100, 50, 200 |
| price | 150.44, 120.92 |
| trader_id | T001, T015 (optional) |

**Sample:**
```csv
timestamp,ticker,action,quantity,price,trader_id
2024-01-15 19:02:00,GOOGL,SELL,100,120.92,T004
2024-01-15 19:05:00,AAPL,BUY,200,150.44,T015
```

## How to Use the Dashboard

1. **Upload Data** - Click "Upload transaction CSV" in sidebar
2. **Load** - Click "Load & Analyze Data" button
3. **Explore** - Check out Overview, Analytics, and Raw Data tabs
4. **Add API Key** - Paste your Gemini key in sidebar
5. **Generate Insights** - Click "Generate Insights" button, optionally customize the prompt
6. **Export** - Download results as CSV or JSON

## Code Structure

```
dashboard.py                 # Streamlit UI (main entry)
transaction_processor.py     # Data loading, cleaning, analytics
insights_generator.py        # LLM API integration
main.py                      # CLI mode
requirements.txt             # Dependencies
.env                         # API key (you create this)
```

## API Reference

### TransactionProcessor

```python
from transaction_processor import TransactionProcessor

# Load and process data
processor = TransactionProcessor('data.csv')
processor.clean_data()
analytics = processor.analyze_data()

# Query data
top_tickers = processor.get_top_tickers(10, 'volume')
top_traders = processor.get_top_traders(5, 'count')

# Export
processor.export_analytics('report.json')
```

### InsightsGenerator

```python
from insights_generator import InsightsGenerator

# Initialize
insights_gen = InsightsGenerator(
    api_key='your_gemini_key',
    provider='gemini'
)

# Generate insights
insights = insights_gen.generate_insights(
    analytics=analytics,
    custom_prompt="Focus on concentration risks"
)

# Save
insights_gen.save_insights(insights, 'insights.json')
```

## Troubleshooting

### "Could not find a valid Gemini model"
- Check your API key is correct
- Verify internet connection
- Check your quota at: https://makersuite.google.com

### "API quota exceeded"
- Gemini free tier has limits (1500 requests/minute)
- Wait a bit before retrying

### Data doesn't load
- Check CSV has all required columns (case-sensitive)
- Verify timestamp format is `YYYY-MM-DD HH:MM:SS` or similar
- Open sample_transactions.csv as reference

## Data Cleaning Decisions

**Missing trader_id (27 rows)** - Grouped all as single "UNKNOWN" trader instead of dropping. Preserves trading volume data while tracking unknown traders as group.

**Duplicates (7 rows)** - Removed. Likely data entry errors.

**Invalid data types** - Converted to numeric/datetime or dropped if conversion failed.

**Missing critical fields** - Dropped rows missing timestamp, ticker, action, quantity, or price.

## Performance

- Loads 1000 transactions in ~100ms
- Generates insights in 5-15 seconds (API dependent)
- Dashboard renders instantly via streamlit

## Files Generated

Running the app creates:
- `temp_transactions.csv` - Uploaded file (temporary)
- `processed_transactions.csv` - Cleaned data
- `analytics_report.json` - Analytics results
- `generated_insights.json` - AI insights

## What's Different

Used **google-genai** SDK (newer, recommended by Google) instead of deprecated google-generativeai.

**Handling unknown traders** - Instead of dropping rows with missing trader_id, I group them all as "UNKNOWN" and treat as single trader. This preserves transaction data.

**Graceful API failures** - If Gemini API fails, don't crash. Generate fallback insights instead. App always works even without API.

Load data → explore → generate insights. No complicated setup, no hidden dependencies. Everything's documented in the code.
