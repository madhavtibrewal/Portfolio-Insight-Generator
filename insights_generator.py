# insights_generator.py
import os
from typing import Dict, List, Optional
import json
from datetime import datetime
from dotenv import load_dotenv

class InsightsGenerator:
    """Generates insights from transaction data using LLM."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-3-flash-preview", provider: str = "gemini"):
        """
        Initialize the InsightsGenerator.
        
        Args:
            api_key (str): API key (OpenAI or Google)
            model (str): LLM model to use
            provider (str): Which provider to use ("openai" or "gemini")
        """
        load_dotenv()
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.provider = provider.lower()
        self.insights_history = []
        self.use_new_sdk = False  # Will be set by _init_gemini
        
        # Initialize appropriate client
        if self.api_key:
            if self.provider == "openai":
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=self.api_key)
                    print(f"OpenAI client initialized with model: {model}")
                except ImportError:
                    print("OpenAI package not installed. Falling back to Gemini.")
                    self.provider = "gemini"
                    self._init_gemini()
                except Exception as e:
                    print(f"Error initializing OpenAI: {e}. Falling back to Gemini.")
                    self.provider = "gemini"
                    self._init_gemini()
            elif self.provider == "gemini":
                self._init_gemini()
            else:
                print(f"Unknown provider: {self.provider}. Using fallback mode.")
                self.client = None
        else:
            self.client = None
            print("No API key provided. AI insights will be disabled.")
    
    def _init_gemini(self):
        """Initialize Google Gemini client using google-genai SDK."""
        # Try new google-genai SDK first
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            
            # Initialize client with API key (don't test during init - test during actual use)
            self.client = genai.Client(api_key=self.api_key)  # type: ignore
            self.genai_types = types  # Store for later use
            self.use_new_sdk = True  # Flag to indicate which SDK we're using
            
            # Set default model fallback list for later use
            self.model_fallbacks = [
                self.model,  # Try the specified model first
                "gemini-3-flash-preview",  # Preview model
                "gemini-2.5-flash",  # Stable alternative
                "gemini-2.5-pro",  # Pro version
            ]
            
            print(f"Google Gemini client initialized (using google-genai SDK). Will use model: {self.model}")
            return
                
        except ImportError:
            print("google-genai package not found.")
            print("Install with: pip install google-genai")
            self.client = None
            self.genai_types = None
            self.use_new_sdk = False
        except Exception as e:
            print(f"Error initializing Gemini: {str(e)}")
            self.client = None
            self.genai_types = None
            self.use_new_sdk = False
    
    def prepare_analytics_summary(self, analytics: Dict) -> str:
        """
        Prepare a text summary from analytics data for LLM consumption.
        
        Args:
            analytics (Dict): Analytics data from TransactionProcessor
            
        Returns:
            str: Formatted text summary
        """
        summary = []
        
        # Basic summary
        stats = analytics.get('summary_stats', {})
        summary.append("PORTFOLIO TRADING DATA SUMMARY")
        summary.append("=" * 50)
        summary.append(f"• Total Transactions: {stats.get('total_transactions', 0):,}")
        summary.append(f"• Unique Tickers: {stats.get('unique_tickers', 0)}")
        summary.append(f"• Unique Traders: {stats.get('unique_traders', 0)}")
        summary.append(f"• Total Volume Traded: {stats.get('total_volume', 0):,.0f} shares")
        summary.append(f"• Total Transaction Value: ${stats.get('total_value', 0):,.2f}")
        
        date_range = stats.get('date_range', {})
        if date_range:
            summary.append(f"• Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
        
        # Top tickers by volume
        volume_data = analytics.get('volume_per_ticker', {})
        if volume_data:
            top_tickers = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)[:10]
            summary.append("\nTOP 10 TICKERS BY TRADING VOLUME:")
            for ticker, volume in top_tickers:
                summary.append(f"  • {ticker}: {volume:,.0f} shares")
        
        # Net positions
        net_positions = analytics.get('net_position', {})
        if net_positions:
            summary.append("\nTOP 10 NET POSITIONS (Buys - Sells):")
            for ticker, position in sorted(net_positions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                action = "NET BUY" if position > 0 else "NET SELL"
                summary.append(f"  • {ticker}: {abs(position):,.0f} shares ({action})")
        
        # Most active traders
        trader_data = analytics.get('trader_activity', {})
        if trader_data:
            summary.append("\nTOP 5 MOST ACTIVE TRADERS:")
            
            # Handle different data structures
            trader_list = []
            for trader_id, stats in trader_data.items():
                if isinstance(stats, dict):
                    count = stats.get('transaction_count', 0)
                else:
                    count = stats
                trader_list.append((trader_id, count))
            
            # Sort by transaction count
            trader_list.sort(key=lambda x: x[1], reverse=True)
            
            for trader_id, count in trader_list[:5]:
                summary.append(f"  • {trader_id}: {count} transactions")
        
        # Time patterns
        hourly_data = analytics.get('hourly_activity', {})
        if hourly_data and 'transaction_count' in hourly_data:
            summary.append("\nTRADING ACTIVITY BY HOUR (Peak Hours):")
            hour_counts = hourly_data['transaction_count']
            # Sort by count
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for hour_str, count in sorted_hours:
                if count > 0:
                    hour = int(hour_str)
                    summary.append(f"  • {hour:02d}:00 - {count} transactions")
        
        # Concentration analysis
        if volume_data:
            volume_values = list(volume_data.values())
            total_volume = sum(volume_values)
            if total_volume > 0:
                top_3_pct = sum(sorted(volume_values, reverse=True)[:3]) / total_volume * 100
                summary.append(f"\nCONCENTRATION: Top 3 tickers account for {top_3_pct:.1f}% of total volume")
        
        return "\n".join(summary)
    
    def generate_insights(self, analytics: Dict, custom_prompt: str = None) -> Dict:
        """
        Generate insights using LLM based on transaction analytics.
        
        Args:
            analytics (Dict): Analytics data from TransactionProcessor
            custom_prompt (str): Optional custom prompt for the LLM
            
        Returns:
            Dict: Generated insights with sections
        """
        if not self.client:
            print("AI client not available. Using fallback insights.")
            return self._generate_fallback_insights(analytics, "AI client not initialized. Please check your API key and ensure the required package is installed.")
        
        # Prepare the data summary
        data_summary = self.prepare_analytics_summary(analytics)
        
        # Define the system prompt - adapt based on whether custom prompt is provided
        if custom_prompt:
            # When custom prompt is provided, focus on answering that specific question
            system_prompt = """You are a senior financial analyst specializing in trading pattern analysis and risk assessment. 
            The user has provided a specific question or focus area for analysis. Your response should:
            1. Directly address the user's specific question or focus area
            2. Be highly specific and data-driven
            3. Use concrete examples from the trading data
            4. Structure your response with clear sections and bullet points
            5. Stay focused on the user's question rather than providing generic analysis
            
            Prioritize answering the user's specific question in detail."""
        else:
            # Default system prompt for general analysis
            system_prompt = """You are a senior financial analyst specializing in trading pattern analysis and risk assessment. 
            Analyze the provided trading data and generate comprehensive insights focusing on:
            1. Trading patterns and behaviors
            2. Concentration risks
            3. Unusual or suspicious activity
            4. Market sentiment indicators
            5. Risk factors and recommendations
            
            Structure your response with clear sections and bullet points.
            Be specific, data-driven, and actionable in your insights."""
        
        # Define the user prompt
        if custom_prompt:
            # Emphasize the custom prompt more strongly
            user_prompt = f"""IMPORTANT: Please focus specifically on answering this question:

{custom_prompt}

Provide a detailed, specific analysis that directly addresses this question. Use concrete data points, numbers, and examples from the trading data to support your analysis.

Here is the trading data:

{data_summary}"""
        else:
            user_prompt = f"""Analyze these trading patterns and provide insights on:
            1. What are the dominant trading patterns?
            2. Are there any concentration risks (certain traders or tickers dominating)?
            3. Flag any unusual trading activity (large orders, rapid trading, etc.)
            4. What does this data suggest about market sentiment?
            5. What risk factors should be monitored?
            
            Here is the trading data:
            
            {data_summary}"""
        
        try:
            print(f"Calling {self.provider.upper()} API with model: {self.model}")
            
            # Call appropriate API based on provider
            if self.provider == "openai":
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                insights_text = response.choices[0].message.content
                
                # Get token usage
                token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                } if hasattr(response, 'usage') and response.usage else {}
                
            elif self.provider == "gemini":
                # Call Google Gemini API - support both new and old SDKs with model fallbacks
                if getattr(self, 'use_new_sdk', False):
                    # New google-genai SDK
                    from google.genai import types  # type: ignore
                    
                    # Try models with fallbacks
                    model_fallbacks = getattr(self, 'model_fallbacks', [self.model])
                    last_error = None
                    response = None
                    
                    for model_name in model_fallbacks:
                        try:
                            response = self.client.models.generate_content(  # type: ignore
                                model=model_name,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(  # type: ignore
                                    system_instruction=system_prompt,
                                    temperature=0.7,
                                    max_output_tokens=1500
                                )
                            )
                            self.model = model_name  # Update to working model
                            break
                        except Exception as model_error:
                            last_error = model_error
                            continue
                    
                    if response is None:
                        error_msg = f"Failed to generate content with any available model. Last error: {last_error}"
                        raise Exception(error_msg)
                    
                    # Try multiple ways to extract text from response
                    insights_text = ""
                    if hasattr(response, 'text') and response.text:  # type: ignore
                        insights_text = response.text  # type: ignore
                    elif hasattr(response, 'candidates') and response.candidates:  # type: ignore
                        # Try to extract from candidates
                        candidate = response.candidates[0]  # type: ignore
                        if hasattr(candidate, 'content') and candidate.content:  # type: ignore
                            content = candidate.content  # type: ignore
                            if hasattr(content, 'parts') and content.parts:  # type: ignore
                                parts = content.parts  # type: ignore
                                text_parts = [part.text for part in parts if hasattr(part, 'text') and part.text]  # type: ignore
                                insights_text = " ".join(text_parts)
                            elif hasattr(content, 'text'):  # type: ignore
                                insights_text = content.text  # type: ignore
                    elif hasattr(response, 'parts') and response.parts:  # type: ignore
                        # Direct parts access
                        text_parts = [part.text for part in response.parts if hasattr(part, 'text') and part.text]  # type: ignore
                        insights_text = " ".join(text_parts)
                    
                    if not insights_text:
                        # Check if response was blocked
                        if hasattr(response, 'prompt_feedback'):  # type: ignore
                            feedback = response.prompt_feedback  # type: ignore
                            if hasattr(feedback, 'block_reason') and feedback.block_reason:  # type: ignore
                                block_reason = feedback.block_reason  # type: ignore
                                raise ValueError(f"Response blocked by safety filters. Reason: {block_reason}")
                        
                        # Provide detailed error information
                        response_info = f"Response type: {type(response)}, "
                        response_info += f"Attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}"
                        raise ValueError(f"Empty response from Gemini API. {response_info}")
                    
                    # Token usage may be available in response.usage_metadata if supported
                    token_usage = {}
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:  # type: ignore
                        usage = response.usage_metadata  # type: ignore
                        token_usage = {
                            'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                            'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                            'total_tokens': getattr(usage, 'total_token_count', 0)
                        }
                else:
                    # Old google-generativeai SDK
                    # If client is the module, create a model instance
                    if hasattr(self.client, 'GenerativeModel'):
                        genai = self.client
                    else:
                        import google.generativeai as genai  # type: ignore
                    
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    
                    # Try models with fallbacks
                    model_fallbacks = getattr(self, 'model_fallbacks', [self.model])
                    last_error = None
                    response = None
                    
                    for model_name in model_fallbacks:
                        try:
                            model = genai.GenerativeModel(model_name)  # type: ignore
                            response = model.generate_content(full_prompt)  # type: ignore
                            self.model = model_name  # Update to working model
                            break
                        except Exception as model_error:
                            last_error = model_error
                            continue
                    
                    if response is None:
                        error_msg = f"Failed to generate content with any available model. Last error: {last_error}"
                        raise Exception(error_msg)
                    
                    # Try multiple ways to extract text from response (old SDK)
                    insights_text = ""
                    if hasattr(response, 'text') and response.text:  # type: ignore
                        insights_text = response.text  # type: ignore
                    elif hasattr(response, 'candidates') and response.candidates:  # type: ignore
                        # Try to extract from candidates
                        candidate = response.candidates[0]  # type: ignore
                        if hasattr(candidate, 'content') and candidate.content:  # type: ignore
                            content = candidate.content  # type: ignore
                            if hasattr(content, 'parts') and content.parts:  # type: ignore
                                parts = content.parts  # type: ignore
                                text_parts = [part.text for part in parts if hasattr(part, 'text') and part.text]  # type: ignore
                                insights_text = " ".join(text_parts)
                            elif hasattr(content, 'text'):  # type: ignore
                                insights_text = content.text  # type: ignore
                    
                    if not insights_text:
                        # Check if response was blocked
                        if hasattr(response, 'prompt_feedback'):  # type: ignore
                            feedback = response.prompt_feedback  # type: ignore
                            if hasattr(feedback, 'block_reason') and feedback.block_reason:  # type: ignore
                                block_reason = feedback.block_reason  # type: ignore
                                raise ValueError(f"Response blocked by safety filters. Reason: {block_reason}")
                        
                        # Provide detailed error information
                        response_info = f"Response type: {type(response)}, "
                        response_info += f"Attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}"
                        raise ValueError(f"Empty response from Gemini API. {response_info}")
                    
                    token_usage = {}
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # Parse the response into structured format
            structured_insights = self._parse_insights_response(insights_text)
            
            # Add metadata
            structured_insights['metadata'] = {
                'provider': self.provider,
                'model': self.model,
                'timestamp': datetime.now().isoformat(),
                'token_usage': token_usage
            }
            
            # Store in history
            self.insights_history.append({
                'timestamp': datetime.now().isoformat(),
                'analytics_summary': analytics.get('summary_stats', {}),
                'insights': structured_insights
            })
            
            print(f"Successfully generated {self.provider.upper()} insights")
            return structured_insights
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating insights from {self.provider}: {error_msg}")
            # Return a fallback response with error details
            return self._generate_fallback_insights(analytics, error_msg)
    
    def _parse_insights_response(self, response_text: str) -> Dict:
        """
        Parse the LLM response into structured sections.
        
        Args:
            response_text (str): Raw LLM response
            
        Returns:
            Dict: Structured insights with sections
        """
        # Simple parsing based on common section headers
        sections = {}
        current_section = None
        current_content = []
        
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if line is a section header (starts with ## or ends with :)
            if line.startswith('## ') or line.startswith('# '):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace('#', '').replace(':', '').strip()
                current_content = []
            elif line.endswith(':') and len(line.split()) < 5:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace(':', '').strip()
                current_content = []
            elif current_section:
                if line:
                    current_content.append(line)
            elif line and not current_section:
                # If we haven't found a section yet, start with "Summary"
                current_section = "Summary"
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no sections were found, treat entire response as one section
        if not sections:
            sections['Analysis'] = response_text
        
        return {
            'raw_response': response_text,
            'sections': sections,
            'is_fallback': False
        }
    
    def _generate_fallback_insights(self, analytics: Dict, error_message: str = None) -> Dict:
        """
        Generate fallback insights when API call fails.
        
        Args:
            analytics (Dict): Analytics data
            error_message (str): Optional error message to include
            
        Returns:
            Dict: Fallback insights
        """
        summary = self.prepare_analytics_summary(analytics)
        
        # Create simple manual insights based on the data
        insights_text = """# Trading Analysis Summary

## Key Observations:
1. **High Concentration**: A few traders and tickers dominate the trading activity
2. **Active Trading**: Multiple transactions occur throughout the day with peaks during market hours
3. **Mixed Sentiment**: Both BUY and SELL actions are present across different tickers

## Risk Factors:
- **Concentration Risk**: Heavy trading in specific stocks could indicate lack of diversification
- **Trader Dependency**: A small number of traders account for majority of volume
- **Market Timing**: Trading occurs across extended hours, which may indicate algorithmic trading

## Recommendations:
1. Monitor top traders for compliance with trading limits
2. Review concentration in heavily traded tickers
3. Implement additional oversight for after-hours trading
4. Consider diversification strategies to reduce concentration risk
"""
        
        # Create status message with error details if available
        if error_message:
            status_msg = f"AI insights unavailable. Error: {error_message}\\n\\nPlease check:\\n- API key is valid and set correctly\\n- Required package is installed (pip install google-genai)\\n- API quota has not been exceeded\\n- Internet connection is active"
        else:
            status_msg = "AI insights disabled or API quota exceeded. Add valid API key to enable AI analysis."
        
        return {
            'raw_response': insights_text,
            'sections': {
                'Trading Analysis Summary': insights_text,
                'Data Overview': summary[:500] + "...",
                'Status': status_msg
            },
            'metadata': {
                'provider': 'fallback',
                'model': 'fallback',
                'timestamp': datetime.now().isoformat(),
                'note': 'Generated without AI API',
                'error': error_message
            },
            'is_fallback': True
        }
    
    def save_insights(self, insights: Dict, file_path: str = 'generated_insights.json'):
        """
        Save generated insights to a JSON file.
        
        Args:
            insights (Dict): Insights data
            file_path (str): Path to save the JSON file
        """
        with open(file_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"Insights saved to {file_path}")