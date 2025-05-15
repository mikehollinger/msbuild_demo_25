# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, io, re
import pandas as pd
import streamlit as st
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from dotenv import load_dotenv
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr


# Load environment variables from .env file
load_dotenv()

# === Configuration ===
api_key = os.environ.get("NVIDIA_API_KEY")
api_url = os.environ.get("API_URL")

# Configure logging with colored log levels
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# ANSI color codes for log levels
LOG_COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m',     # Reset to default
}

# Custom formatter that adds color only to the log level
class ColoredLevelFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            colored_levelname = f"{LOG_COLORS[levelname]}{levelname}{LOG_COLORS['RESET']}"
            record.levelname = colored_levelname
        return super().format(record)

# Configure logging with the custom formatter
formatter = ColoredLevelFormatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level, logging.INFO))
# Remove existing handlers to avoid duplicates
for hdlr in root_logger.handlers[:]:
    root_logger.removeHandler(hdlr)
root_logger.addHandler(handler)

logger = logging.getLogger(__name__)

# Suppress httpcore debug logs
logging.getLogger('httpcore').setLevel(logging.WARNING)

client = OpenAI(
  base_url = api_url,
  api_key = api_key
)

# === Unified API Call Function =====================================
def call_llm_api(
    prompt: str, 
    system_content: str = "detailed thinking off.", 
    stream: bool = False, 
    temperature: float = 0.2, 
    max_tokens: int = 4096, 
    thinking_placeholder: Optional[Any] = None,
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    thinking_title: str = "Model Thinking"
) -> Union[str, Tuple[str, str]]:
    """
    Unified function to call the LLM API with consistent handling of streaming and thinking tags.
    
    Args:
        prompt: The user prompt to send to the LLM
        system_content: System prompt to control LLM behavior
        stream: Whether to stream the response
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        thinking_placeholder: Streamlit placeholder for displaying thinking (only used if stream=True)
        model: Model to use for inference
        thinking_title: Title to display in the thinking section
        
    Returns:
        If stream=False: Just the response content with thinking tags removed
        If stream=True: A tuple of (thinking_content, final_response)
    """
    logger.debug(f"Calling LLM API with prompt: {prompt}")
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    if not stream:
        # Non-streaming call
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            logger.debug(f"API non-streaming response: {result}")
            
            # Extract thinking from non-streaming response too
            thinking_content = ""
            cleaned_response = result
            
            # Extract thinking tags and clean response
            if "<think>" in result:
                # Extract content inside thinking tags
                thinking_match = re.search(r"<think>(.*?)</think>", result, re.DOTALL)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()
                    
                # Remove thinking tags and their contents
                cleaned_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
                logger.debug(f"Extracted thinking content: {thinking_content[:100]}...")
            
            # Return only the cleaned response without thinking tags
            return cleaned_response
        except Exception as exc:
            error_msg = f"Error calling LLM API: {exc}"
            logger.error(error_msg)
            return error_msg
    else:
        # Streaming call with thinking tag extraction
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            thinking_content = ""
            in_think = False
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    
                    # Extract thinking tags as they stream
                    if "<think>" in token:
                        in_think = True
                        token = token.split("<think>", 1)[1]
                    if "</think>" in token:
                        token = token.split("</think>", 1)[0]
                        in_think = False
                    if in_think or ("<think>" in full_response and not "</think>" in full_response):
                        thinking_content += token
                        if thinking_placeholder:
                            thinking_placeholder.markdown(
                                f'<details class="thinking" open><summary>🤔 {thinking_title}</summary><pre>{thinking_content}</pre></details>',
                                unsafe_allow_html=True
                            )
            
            logger.debug(f"API streaming response completed, thinking content length: {len(thinking_content)}")
            
            # After streaming, extract final reasoning (outside <think>...</think>)
            cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
            return thinking_content, cleaned
        except Exception as exc:
            error_msg = f"Error in streaming LLM API call: {exc}"
            logger.error(error_msg)
            return "", error_msg

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    # First, remove any thinking tags and their content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Then extract code block
    start = text.find("```python")
    if start == -1:
        # Try without python specification (just ```)
        start = text.find("```")
        if start == -1:
            return ""
        start += len("```")
    else:
        start += len("```python")
    
    end = text.find("```", start)
    if end == -1:
        return ""
        
    return text[start:end].strip()

def get_system_prompt(base_prompt: str) -> str:
    """Builds a system prompt with reasoning enabled/disabled based on session state."""
    reasoning_enabled = st.session_state.get("reasoning_enabled", True)
    reasoning_prefix = "detailed thinking on." if reasoning_enabled else "detailed thinking off."
    return f"{reasoning_prefix} {base_prompt}"

# === CodeGeneration TOOLS ============================================

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    base_prompt = "You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."
    system_content = get_system_prompt(base_prompt)
    
    prompt = query
    
    logger.debug(f"Calling QueryUnderstandingTool with query: {query}")
    
    result = call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=False,
        temperature=0.1,
        max_tokens=5
    )
    
    # Extract the response and convert to boolean
    # Strip any whitespace and convert to lowercase for comparison
    intent_response = result.strip().lower()
    logger.info(f"Query understanding result: {intent_response} for query: {query}")
    
    # Check if the response contains "true" anywhere (handles partial responses)
    return "true" in intent_response

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+plotly code for a plot based on the query and columns."""
    logger.debug(f"Generating plot code prompt for query: {query}")
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and Plotly** to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and plotly.express (as px) or plotly.graph_objects (as go) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* Plotly Figure) to a variable named `result`.
    3. Create only ONE relevant plot. Add descriptive title and axis labels.
    4. For better Streamlit integration, use Plotly's update_layout() method to set plot size and margins.
    5. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    6. IMPORTANT: Do NOT include any import statements. The variables 'pd', 'df', 'px', and 'go' are already defined in the execution environment.
    """

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    logger.debug(f"Generating data analysis code prompt for query: {query}")
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    4. IMPORTANT: Do NOT include any import statements. The variables 'pd' and 'df' are already defined in the execution environment.
    """

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame, max_retries: int = 3, thinking_placeholder: Optional[Any] = None):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    logger.info(f"CodeGenerationAgent processing query: {query}")
    should_plot = QueryUnderstandingTool(query)
    
    # If the query starts with "plot" or contains visualization keywords, force plot mode
    if query.lower().startswith("plot") or "visualize" in query.lower() or "chart" in query.lower() or "graph" in query.lower():
        should_plot = True
        logger.info(f"Forcing plot mode based on query keywords")
        
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)
    
    code = ""
    error_msg = ""
    thinking_content = ""
    retries = 0
    
    base_prompt = "You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis. IMPORTANT: Do not include any import statements in your code. Variables 'pd', 'df', 'px', and 'go' are already available in the execution environment."
    system_content = get_system_prompt(base_prompt)
    
    while retries <= max_retries:
        # If this is a retry, include the error message in the prompt
        retry_prompt = prompt
        if retries > 0:
            retry_prompt = f"""
            {prompt}
            
            The previous code generated an error:
            {error_msg}
            
            Here is the previous code attempt that failed:
            ```python
            {code}
            ```
            
            Please fix the code to avoid this error.
            """
            logger.info(f"Retrying code generation (attempt {retries}/{max_retries}) after error: {error_msg}")
        
        # Create a placeholder for thinking output
        current_thinking_placeholder = thinking_placeholder or st.empty()
        
        # Use streaming API call to show thinking in real-time
        current_thinking, result = call_llm_api(
            prompt=retry_prompt,
            system_content=system_content,
            stream=True,
            temperature=0.2,
            max_tokens=8192,
            thinking_placeholder=current_thinking_placeholder,
            thinking_title="Code Generation Thinking"
        )
        
        # Save the thinking content
        thinking_content = current_thinking

        code = extract_first_code_block(result)
        logger.debug(f"Generated code: {code}")
        
        # Try executing the code
        if code:
            # Test execution to see if it works
            result = ExecutionAgent(code, df, should_plot)
            # Check if result is an error message
            if isinstance(result, str) and result.startswith("Error executing code"):
                error_msg = result
                retries += 1
                continue
            else:
                # Code executed successfully
                return code, should_plot, error_msg, thinking_content
        
        retries += 1
    
    # If we've reached max retries and still have errors, return the last code anyway
    logger.warning(f"Reached maximum retries ({max_retries}) with errors, returning last generated code")
    return code, should_plot, error_msg, thinking_content

# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    logger.debug(f"Executing generated code:\n{code}")
    
    env = {"pd": pd, "df": df}
    if should_plot:
        env["px"] = px
        env["go"] = go
        env["io"] = io
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, {}, env)
        
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        if stdout_content:
            logger.debug(f"Code execution stdout:\n{stdout_content}")
        if stderr_content:
            logger.debug(f"Code execution stderr:\n{stderr_content}")
            
        result = env.get("result", None)
        return result
    except Exception as exc:
        error_msg = f"Error executing code: {exc}"
        logger.error(error_msg)
        
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        if stdout_content:
            logger.debug(f"Code execution stdout before error:\n{stdout_content}")
        if stderr_content:
            logger.debug(f"Code execution stderr before error:\n{stderr_content}")
            
        return error_msg

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any, code: str = "") -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    logger.debug("Building reasoning prompt")
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = 'plotly' in str(type(result))

    if is_error:
        desc = result
    elif is_plot:
        # Extract information from Plotly figure
        title = ""
        data_summary = ""
        
        try:
            # Get title from layout
            if hasattr(result, 'layout') and hasattr(result.layout, 'title'):
                if hasattr(result.layout.title, 'text'):
                    title = result.layout.title.text
                else:
                    title = str(result.layout.title)
            
            # Extract data from the figure
            if hasattr(result, 'data') and result.data:
                # Determine chart type from first trace
                trace = result.data[0]
                trace_type = trace.type if hasattr(trace, 'type') else "unknown"
                
                # Get axis titles
                x_title = result.layout.xaxis.title.text if hasattr(result.layout, 'xaxis') and hasattr(result.layout.xaxis, 'title') else "x"
                y_title = result.layout.yaxis.title.text if hasattr(result.layout, 'yaxis') and hasattr(result.layout.yaxis, 'title') else "y"
                
                # Check if we have multiple traces (for grouped/stacked bars, multiple lines, etc)
                trace_count = len(result.data)
                has_multiple_traces = trace_count > 1
                
                # Extract series names/colors if multiple traces
                series_names = []
                if has_multiple_traces:
                    for t in result.data:
                        if hasattr(t, 'name') and t.name:
                            series_names.append(t.name)
                
                # Extract some data points based on chart type
                if trace_type == "bar":
                    if has_multiple_traces:
                        # For grouped/stacked bar charts
                        categories = trace.x[:5] if hasattr(trace, 'x') else []
                        chart_subtype = "grouped" if hasattr(result.layout, 'barmode') and result.layout.barmode == "group" else "stacked"
                        
                        # Extract sample data for each series
                        series_data = []
                        for i, t in enumerate(result.data[:3]):  # Limit to first 3 series for readability
                            if hasattr(t, 'name') and hasattr(t, 'y'):
                                series_name = t.name
                                series_values = t.y[:5] if len(t.y) > 0 else []
                                series_data.append(f"{series_name}: {series_values}")
                        
                        data_summary = f"Chart type: {chart_subtype.capitalize()} bar chart\nCategories: {categories}\nSeries values:\n" + "\n".join(series_data)
                    else:
                        # Simple bar chart
                        x_data = trace.x[:5] if hasattr(trace, 'x') else []
                        y_data = trace.y[:5] if hasattr(trace, 'y') else []
                        data_summary = f"Chart type: Bar chart\nCategories: {x_data}\nValues: {y_data}"
                
                elif trace_type == "scatter":
                    x_data = trace.x[:5] if hasattr(trace, 'x') else []
                    y_data = trace.y[:5] if hasattr(trace, 'y') else []
                    name = trace.name if hasattr(trace, 'name') else ""
                    mode = trace.mode if hasattr(trace, 'mode') else ""
                    
                    if has_multiple_traces:
                        # Extract sample data for each series
                        series_data = []
                        for i, t in enumerate(result.data[:3]):  # Limit to first 3 series for readability
                            if hasattr(t, 'name') and hasattr(t, 'x') and hasattr(t, 'y'):
                                series_name = t.name
                                x_vals = t.x[:3] if len(t.x) > 0 else []
                                y_vals = t.y[:3] if len(t.y) > 0 else []
                                points = list(zip(x_vals, y_vals))
                                series_data.append(f"{series_name}: {points}")
                        
                        data_summary = f"Chart type: Multi-series scatter plot ({mode})\nSeries values:\n" + "\n".join(series_data)
                    else:
                        data_summary = f"Chart type: Scatter ({mode})\nSeries: {name}\nSample points: {list(zip(x_data, y_data))}"
                
                elif trace_type == "pie":
                    labels = trace.labels[:5] if hasattr(trace, 'labels') else []
                    values = trace.values[:5] if hasattr(trace, 'values') else []
                    data_summary = f"Chart type: Pie chart\nCategories: {labels}\nValues: {values}"
                
                else:
                    # Generic extraction for other chart types
                    data_summary = f"Chart type: {trace_type}\n"
                    if has_multiple_traces:
                        data_summary += f"Number of series: {trace_count}\n"
                        if series_names:
                            data_summary += f"Series names: {series_names}\n"
                    
                    for attr in ['x', 'y', 'z', 'values', 'labels']:
                        if hasattr(trace, attr):
                            attr_data = getattr(trace, attr)
                            if attr_data and len(attr_data) > 0:
                                data_summary += f"{attr}: {attr_data[:5]}\n"
                
                data_summary += f"\nAxes: {x_title} vs {y_title}"
                
        except Exception as e:
            logger.warning(f"Error extracting Plotly figure info: {e}")
            data_summary = "Plot details could not be extracted"
        
        desc = f"[Plot: {title or 'Chart'}]\n{data_summary}"
    elif isinstance(result, pd.DataFrame):
        # For DataFrame results, include shape and sample data
        row_count = len(result)
        col_count = len(result.columns)
        
        if row_count > 0:
            # Include descriptive statistics if available
            if all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns if col in result):
                desc = f"DataFrame({row_count}×{col_count}):\nSummary statistics:\n{result.describe().to_string()}"
            else:
                # Include sample rows
                sample_size = min(5, row_count)
                desc = f"DataFrame({row_count}×{col_count}):\nSample rows:\n{result.head(sample_size).to_string()}"
        else:
            desc = f"Empty DataFrame with {col_count} columns"
    elif isinstance(result, pd.Series):
        # For Series results, include stats
        if pd.api.types.is_numeric_dtype(result):
            desc = f"Series({len(result)}):\n{result.describe().to_string()}"
        else:
            # For categorical series, show value counts
            if hasattr(result, 'value_counts') and callable(getattr(result, 'value_counts')):
                counts = result.value_counts()
                if len(counts) <= 10:  # Show all if not too many categories
                    desc = f"Series({len(result)}) value counts:\n{counts.to_string()}"
                else:
                    desc = f"Series({len(result)}) top value counts:\n{counts.head(5).to_string()}"
            else:
                desc = f"Series({len(result)}):\n{result.head(5).to_string()}"
                if len(result) > 5:
                    desc += f"\n... and {len(result)-5} more"
    else:
        # For scalar or other results
        desc = str(result)[:300]
        if len(str(result)) > 300:
            desc += "..."

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        The code used to generate the result:
        ```python
        {code}
        ```
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows and its key insights.'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The code used to generate the result:
        ```python
        {code}
        ```
        The result is:
        {desc}
        Explain in 2–3 concise sentences what this tells about the data.'''
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any, code: str = "", thinking_placeholder: Optional[Any] = None):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    logger.info("Generating reasoning about results")
    prompt = ReasoningCurator(query, result, code)
    logger.info(f"Reasoning prompt: {prompt}")

    # Get the system prompt with reasoning status
    base_prompt = "You are an insightful data analyst."
    system_content = get_system_prompt(base_prompt)

    # Use the unified API calling function with streaming
    thinking_placeholder = thinking_placeholder or st.empty()
    
    thinking_content, cleaned = call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=True,
        temperature=0.2,
        max_tokens=8192,
        thinking_placeholder=thinking_placeholder,
        thinking_title="Result Analysis Thinking"
    )
    
    logger.info(f"Completed streaming response, thinking content length: {len(thinking_content)}")
    logger.debug(f"Model thinking: {thinking_content}")
    
    return thinking_content, cleaned

# === DataFrameSummary TOOL (pandas only) =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    logger.debug(f"Generating DataFrame summary for shape: {df.shape}")
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt

# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    logger.info(f"Generating dataset insights for DataFrame with shape: {df.shape}")
    prompt = DataFrameSummaryTool(df)
    
    base_prompt = "You are a data analyst providing brief, focused insights."
    system_content = get_system_prompt(base_prompt)
    
    return call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=False,
        temperature=0.2,
        max_tokens=8192
    )

# === Main Streamlit App ===============================================

def main():
    logger.info("Starting Data Analysis Agent application")
    st.set_page_config(layout="wide")
    if "reasoning_enabled" not in st.session_state:
        st.session_state.reasoning_enabled = True  # Default to enabled

    # Add CSS for thinking and code sections
    st.markdown("""
    <style>
    details.thinking {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 5px solid #9e9ac8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    details.thinking summary {
        cursor: pointer;
        font-weight: bold;
        color: #555;
        margin-bottom: 8px;
    }
    details.thinking pre {
        white-space: pre-wrap;
        margin-top: 8px;
        font-size: 0.9em;
        line-height: 1.4;
        background-color: #ffffff;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #e6e6e6;
    }
    details.code {
        background-color: #f6f6f6;
        padding: 12px;
        border-radius: 8px;
        margin-top: 15px;
        border-left: 5px solid #4a90e2;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    details.code summary {
        cursor: pointer;
        font-weight: bold;
        color: #333;
        margin-bottom: 8px;
    }
    details.code pre {
        background-color: #ffffff;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #e6e6e6;
    }
    .thinking-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([3,7])

    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by Azure AI Foundry NVIDIA Llama-3.1-Nemotron-Super-49</a></medium>", unsafe_allow_html=True)
        
        # Add the toggle switch
        reasoning_enabled = st.toggle("Enable Detailed Reasoning", value=st.session_state.reasoning_enabled)
        st.session_state.reasoning_enabled = reasoning_enabled
        
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                logger.info(f"Loading new file: {file.name}")
                st.session_state.df = pd.read_csv(file)
                st.session_state.current_file = file.name
                st.session_state.messages = []
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = DataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df.head())
            st.markdown("### Dataset Insights")
            st.markdown(st.session_state.insights)
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("figure") is not None:
                        # Display Plotly figure directly
                        st.plotly_chart(msg["figure"], use_container_width=True)

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about your data…"):
                logger.info(f"Received user query: {user_q}")
                st.session_state.messages.append({"role": "user", "content": user_q})
                
                # Create containers for thinking output that won't be overwritten
                thinking_container = st.container()
                with thinking_container:
                    st.markdown("#### Model Thinking Process")
                    code_thinking_placeholder = st.empty()
                    st.markdown("---")  # Separator between thinking blocks
                    reasoning_thinking_placeholder = st.empty()
                
                with st.spinner("Working …"):
                    # Pass the code thinking placeholder to CodeGenerationAgent
                    code, should_plot_flag, error_msg, thinking_content = CodeGenerationAgent(
                        user_q, 
                        st.session_state.df,
                        thinking_placeholder=code_thinking_placeholder
                    )
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    
                    # Pass the reasoning thinking placeholder to ReasoningAgent
                    raw_thinking, reasoning_txt = ReasoningAgent(
                        user_q, 
                        result_obj, 
                        code,
                        thinking_placeholder=reasoning_thinking_placeholder
                    )
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build assistant response
                is_plot = 'plotly' in str(type(result_obj))
                figure = None
                if is_plot:
                    logger.debug("Storing generated Plotly figure")
                    figure = result_obj
                    header = "Here is the visualization you requested:"
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                else:
                    header = f"Result: {result_obj}"

                # Show code generation thinking in Model Thinking section
                code_thinking_html = ""
                if thinking_content:
                    code_thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧮 Code Generation Process</summary>'
                        f'<pre>{thinking_content}</pre>'
                        '</details>'
                    )

                # Show reasoning thinking in Model Thinking section
                reasoning_thinking_html = ""
                if raw_thinking:
                    reasoning_thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Result Analysis Process</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Add retry information if there were retries
                if error_msg:
                    explanation_html += f"\n\n<small><em>Note: Some code errors were fixed during generation.</em></small>"

                # Code accordion with proper HTML <pre><code> syntax highlighting
                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                
                # Combine thinking sections first, then explanation and code accordion
                thinking_html = ""
                if code_thinking_html or reasoning_thinking_html:
                    thinking_html = f"{code_thinking_html}{reasoning_thinking_html}"
                
                assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                logger.debug("Adding assistant response to session state")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "figure": figure
                })
                st.rerun()

if __name__ == "__main__":
    main() 