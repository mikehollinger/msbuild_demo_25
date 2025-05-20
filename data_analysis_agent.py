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
import tiktoken  # Add tiktoken for token counting
import traceback  # Import traceback for error line information
import html

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


# Get rumination detection threshold from environment or use default
MAX_THINKING_CHARS = int(os.environ.get("MAX_THINKING_CHARS", "16000"))
logger.info(f"Rumination detection threshold set to {MAX_THINKING_CHARS} characters")


# Get seed value from environment variable or use a default
SEED_VALUE = int(os.environ.get("LLM_SEED", "42"))
logger = logging.getLogger(__name__)
logger.info(f"Using seed value {SEED_VALUE} for LLM API calls")


client = OpenAI(
  base_url = api_url,
  api_key = api_key
)

# === Unified API Call Function =====================================
def call_llm_api(
    prompt: str, 
    system_content: str = "detailed thinking off.", 
    stream: bool = False, 
    temperature: float = 0.0, 
    max_tokens: int = 4096, 
    thinking_placeholder: Optional[Any] = None,
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    thinking_title: str = "Model Thinking",
    max_thinking_chars: int = None,
    top_p: float = 1.0
) -> Union[str, Tuple[str, str], Tuple[str, str, bool], Tuple[str, Dict[str, int]], Tuple[str, str, bool, Dict[str, int]]]:
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
        max_thinking_chars: Maximum characters to allow in thinking before detecting rumination
        top_p: Nucleus sampling parameter (1.0 means no nucleus sampling filter)
        
    Returns:
        If stream=False: A tuple of (response_content, token_counts)
        If stream=True: A tuple of (thinking_content, final_response, ruminated, token_counts)
        Where token_counts is a dictionary with input_tokens, output_tokens, and total_tokens
    """
    # Use the global variable if max_thinking_chars is not provided
    if max_thinking_chars is None:
        max_thinking_chars = MAX_THINKING_CHARS
        
    logger.info(f"Calling LLM API with prompt: {prompt}")
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    # Initialize token counts dictionary
    token_counts = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    # Get encoding based on model
    try:
        if "llama" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 encoding as approximation
        else:
            encoding = tiktoken.encoding_for_model(model)
    except:
        # Fallback to cl100k_base encoding if model-specific encoding not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Count input tokens
    for message in messages:
        input_tokens = len(encoding.encode(message["content"]))
        token_counts["input_tokens"] += input_tokens
    
    token_counts["total_tokens"] = token_counts["input_tokens"]
    logger.debug(f"Input tokens: {token_counts['input_tokens']}")
    
    if not stream:
        # Non-streaming call
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=SEED_VALUE,
                top_p=top_p
            )
            result = response.choices[0].message.content
            logger.debug(f"API non-streaming response: {result}")
            
            # Extract thinking from non-streaming response too
            thinking_content = ""
            cleaned_response = result
            
            # Count output tokens
            token_counts["output_tokens"] = len(encoding.encode(result))
            token_counts["total_tokens"] = token_counts["input_tokens"] + token_counts["output_tokens"]
            logger.info(f"Total tokens used: {token_counts['total_tokens']} (Input: {token_counts['input_tokens']}, Output: {token_counts['output_tokens']})")
            
            # Extract thinking tags and clean response
            if "<think>" in result:
                # Extract content inside thinking tags
                thinking_match = re.search(r"<think>(.*?)</think>", result, re.DOTALL)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()
                    
                # Remove thinking tags and their contents
                cleaned_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
                logger.debug(f"Extracted thinking content: {thinking_content[:100]}...")
            
            # Return cleaned response and token counts
            return cleaned_response, token_counts
        except Exception as exc:
            error_msg = f"Error calling LLM API: {exc}"
            logger.error(error_msg)
            return error_msg, token_counts
    else:
        # Streaming call with thinking tag extraction and token counting
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                seed=SEED_VALUE,
                top_p=top_p
            )
            
            full_response = ""
            thinking_content = ""
            in_think = False
            ruminated = False
            output_tokens = 0
            
            # Create a token count display in the UI if a placeholder is provided
            token_display = st.empty() if thinking_placeholder else None
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    
                    # Count tokens in this chunk
                    chunk_tokens = len(encoding.encode(token))
                    output_tokens += chunk_tokens
                    token_counts["output_tokens"] = output_tokens
                    token_counts["total_tokens"] = token_counts["input_tokens"] + output_tokens
                    
                    # Update token count display
                    if token_display:
                        token_display.markdown(f"Tokens used: {token_counts['total_tokens']} (Input: {token_counts['input_tokens']}, Output: {output_tokens})")
                    
                    # Extract thinking tags as they stream
                    if "<think>" in token:
                        in_think = True
                        token = token.split("<think>", 1)[1]
                    if "</think>" in token:
                        token = token.split("</think>", 1)[0]
                        in_think = False
                    if in_think or ("<think>" in full_response and not "</think>" in full_response):
                        thinking_content += token
                        
                        # Check for rumination - if thinking content exceeds the max length
                        if len(thinking_content) > max_thinking_chars:
                            logger.warning(f"Detected model rumination (thinking content: {len(thinking_content)} chars)")
                            ruminated = True
                            break  # Stop processing the stream
                            
                        if thinking_placeholder:
                            thinking_placeholder.markdown(
                                f'<details class="thinking" open><summary>🤔 {thinking_title}</summary><pre>{thinking_content}</pre></details>',
                                unsafe_allow_html=True
                            )
            
            logger.debug(f"API streaming response completed, thinking content length: {len(thinking_content)}")
            logger.info(f"Final token usage: {token_counts}")
            
            # After streaming, extract final reasoning (outside <think>...</think>)
            cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
            return thinking_content, cleaned, ruminated, token_counts
        except Exception as exc:
            error_msg = f"Error in streaming LLM API call: {exc}"
            logger.error(error_msg)
            return "", error_msg, False, token_counts

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
    base_prompt = "You are an assistant that determines if a query is requesting a data visualization or data that is a simple list of key/value pairs. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."
    # Explicitly disable thinking mode for this API call to ensure we get a usable response with max_tokens=5
    system_content = "detailed thinking off. " + base_prompt
    
    prompt = query
    
    logger.debug(f"Calling QueryUnderstandingTool with query: {query}")
    
    result, token_counts = call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=False,
        max_tokens=5
    )
    
    logger.debug(f"QueryUnderstandingTool token usage: {token_counts}")
    
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
    
    # Get data types and sample data
    df_sample = st.session_state.df.head(3)
    total_rows = len(st.session_state.df)
    dtypes_info = {col: str(st.session_state.df[col].dtype) for col in cols}
    dtypes_str = ", ".join([f"{col} ({dtype})" for col, dtype in dtypes_info.items()])
    
    # Create sample data string
    sample_rows = df_sample.to_string(index=False)
    
    return f"""
    Given DataFrame `df` with {total_rows} total rows and columns:
    
    COLUMNS AND TYPES: {dtypes_str}
    
    SAMPLE DATA:
    {sample_rows}
    
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
    7. When building boolean masks, use `&`, `|`, `~` **with parentheses around every comparison**; never use `and`/`or` on Series.
    8. Never pass a Series to `if`, `while`, or `break` conditions—reduce with `.any()` / `.all()` / `.empty` instead.
    9. Handle edge cases to avoid NaN results:
       - When using groupby with aggregations like std(), var(), etc., ensure there are at least 2 values per group
       - Check with .count() if needed and filter groups with sufficient data
       - For variance/standard deviation, consider using .agg(['mean', 'std']) to provide context
       - Add appropriate filters to ensure valid calculations before plotting
    
    The first triple back-tick after this sentence must open your final code block.

    Begin.
    """

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    logger.debug(f"Generating data analysis code prompt for query: {query}")
    
    # Get data types and sample data
    df_sample = st.session_state.df.head(3)
    total_rows = len(st.session_state.df)
    dtypes_info = {col: str(st.session_state.df[col].dtype) for col in cols}
    dtypes_str = ", ".join([f"{col} ({dtype})" for col, dtype in dtypes_info.items()])
    
    # Create sample data string
    sample_rows = df_sample.to_string(index=False)
    
    return f"""
    Given DataFrame `df` with {total_rows} total rows and columns:
    
    COLUMNS AND TYPES: {dtypes_str}
    
    SAMPLE DATA:
    {sample_rows}
    
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    4. IMPORTANT: Do NOT include any import statements. The variables 'pd' and 'df' are already defined in the execution environment.
    5. Handle edge cases to avoid NaN results:
       - When using groupby with aggregations like std(), var(), etc., ensure there are at least 2 values per group
       - Check with .count() if needed and filter groups with sufficient data
       - For variance/standard deviation, consider using .agg(['mean', 'std']) to provide context
       - Add appropriate filters to ensure valid calculations
    """

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame, max_retries: int = 1, thinking_placeholder: Optional[Any] = None):
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
    ruminated_once = False
    total_tokens_used = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    base_prompt = "You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis. IMPORTANT: Do not include any import statements in your code. Variables 'pd', 'df', 'px', and 'go' are already available in the execution environment. If you use <think> tags for your reasoning, you MUST always ensure your final response includes a ```python code block OUTSIDE of the thinking tags."
    system_content = get_system_prompt(base_prompt)
    
    # Create a token counter display
    token_counter = st.empty()
    
    while retries <= max_retries:
        # If this is a retry, include the error message or rumination issue in the prompt
        retry_prompt = prompt
        if retries > 0:
            if ruminated_once:
                # Get the last 1000 characters of the thinking content as context
                last_thinking = thinking_content[-2000:] if len(thinking_content) > 2000 else thinking_content
                retry_prompt = f"""
                {prompt}
                
                In a previous attempt, you were stuck in a loop of extended thinking without providing a solution.
                Here's the end of your previous thinking:
                
                {last_thinking}
                
                Please be more direct and efficient in your approach. Focus on completing the task as succinctly as possible.
                """
                logger.info(f"Retrying after rumination detection (attempt {retries}/{max_retries})")
                ruminated_once = False  # Reset for this attempt
            elif error_msg:
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
        current_thinking, result, ruminated, token_counts = call_llm_api(
            prompt=retry_prompt,
            system_content=system_content,
            stream=True,
            max_tokens=8192,
            thinking_placeholder=current_thinking_placeholder,
            thinking_title="Code Generation Thinking"
        )
        
        # Update total tokens used
        total_tokens_used["input_tokens"] += token_counts["input_tokens"]
        total_tokens_used["output_tokens"] += token_counts["output_tokens"]
        total_tokens_used["total_tokens"] += token_counts["total_tokens"]
        
        # Update token counter display
        token_counter.markdown(f"**Total tokens used**: {total_tokens_used['total_tokens']} (Input: {total_tokens_used['input_tokens']}, Output: {total_tokens_used['output_tokens']})")
        
        # Save the thinking content
        thinking_content = current_thinking

        # Check if rumination was detected
        if ruminated:
            logger.warning("Model rumination detected, will retry with truncated thinking prompt")
            ruminated_once = True
            retries += 1
            continue

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
                return code, should_plot, error_msg, thinking_content, total_tokens_used
        
        retries += 1
    
    # If we've reached max retries and still have errors, return the last code anyway
    logger.warning(f"Reached maximum retries ({max_retries}) with errors, returning last generated code")
    return code, should_plot, error_msg, thinking_content, total_tokens_used

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
            exec(code, env, env)
        
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        if stdout_content:
            logger.debug(f"Code execution stdout:\n{stdout_content}")
        if stderr_content:
            logger.debug(f"Code execution stderr:\n{stderr_content}")
            
        result = env.get("result", None)
        
        # Check for NaN values in the result
        if isinstance(result, pd.DataFrame) and result.isna().any().any():
            nan_percentage = result.isna().mean().mean() * 100
            # Only flag as error if significant NaN presence
            if nan_percentage > 25:  # If more than 25% of the values are NaN
                logger.warning(f"Result contains {nan_percentage:.1f}% NaN values")
                error_msg = f"Error executing code: Result contains {nan_percentage:.1f}% NaN values. This likely means there's insufficient data for the calculation (e.g., trying to calculate standard deviation with only one value per group). Please revise the approach."
                return error_msg
        elif isinstance(result, pd.Series) and result.isna().any():
            nan_percentage = result.isna().mean() * 100
            if nan_percentage > 25:  # If more than 25% of the values are NaN
                logger.warning(f"Result contains {nan_percentage:.1f}% NaN values")
                error_msg = f"Error executing code: Result contains {nan_percentage:.1f}% NaN values. This likely means there's insufficient data for the calculation. Please revise the approach."
                return error_msg
        
        return result
    except Exception as exc:
        # Get traceback info including line number
        tb = traceback.extract_tb(sys.exc_info()[2])
        # Find the error line in the generated code
        error_line_num = None
        error_line_text = None
        error_context = []
        
        # Split code into lines for context
        code_lines = code.split('\n')
        
        for frame in tb:
            # Check if the traceback is from our generated code
            if frame.filename == '<string>':
                # Line numbers in tracebacks are 1-indexed
                error_line_num = frame.lineno
                if 0 < error_line_num <= len(code_lines):
                    error_line_text = code_lines[error_line_num-1]
                    
                    # Get context: a few lines before and after the error
                    start_line = max(0, error_line_num-3)
                    end_line = min(len(code_lines), error_line_num+2)
                    
                    for i in range(start_line, end_line):
                        prefix = "→ " if i+1 == error_line_num else "  "
                        line_num = i+1
                        error_context.append(f"{prefix}{line_num}: {code_lines[i]}")
                    
                    break
        
        error_msg = f"Error executing code: {exc}"
        if error_line_num:
            error_msg = f"Error executing code at line {error_line_num}: {exc}\n\nError context:\n" + "\n".join(error_context)
        
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
                        categories = trace.x[:50] if hasattr(trace, 'x') else []
                        chart_subtype = "grouped" if hasattr(result.layout, 'barmode') and result.layout.barmode == "group" else "stacked"
                        
                        # Extract sample data for each series
                        series_data = []
                        for i, t in enumerate(result.data[:5]):  # Limit to first 5 series for readability
                            if hasattr(t, 'name') and hasattr(t, 'y'):
                                series_name = t.name
                                series_values = t.y[:50] if len(t.y) > 0 else []
                                series_data.append(f"{series_name}: {series_values}")
                        
                        data_summary = f"Chart type: {chart_subtype.capitalize()} bar chart\nCategories: {categories}\nSeries values:\n" + "\n".join(series_data)
                    else:
                        # Simple bar chart
                        x_data = trace.x[:50] if hasattr(trace, 'x') else []
                        y_data = trace.y[:50] if hasattr(trace, 'y') else []
                        data_summary = f"Chart type: Bar chart\nCategories: {x_data}\nValues: {y_data}"
                
                elif trace_type == "scatter":
                    x_data = trace.x[:50] if hasattr(trace, 'x') else []
                    y_data = trace.y[:50] if hasattr(trace, 'y') else []
                    name = trace.name if hasattr(trace, 'name') else ""
                    mode = trace.mode if hasattr(trace, 'mode') else ""
                    
                    if has_multiple_traces:
                        # Extract sample data for each series
                        series_data = []
                        for i, t in enumerate(result.data[:5]):  # Limit to first 5 series for readability
                            if hasattr(t, 'name') and hasattr(t, 'x') and hasattr(t, 'y'):
                                series_name = t.name
                                x_vals = t.x[:50] if len(t.x) > 0 else []
                                y_vals = t.y[:50] if len(t.y) > 0 else []
                                points = list(zip(x_vals, y_vals))
                                series_data.append(f"{series_name}: {points}")
                        
                        data_summary = f"Chart type: Multi-series scatter plot ({mode})\nSeries values:\n" + "\n".join(series_data)
                    else:
                        data_summary = f"Chart type: Scatter ({mode})\nSeries: {name}\nSample points: {list(zip(x_data, y_data))}"
                
                elif trace_type == "pie":
                    labels = trace.labels[:50] if hasattr(trace, 'labels') else []
                    values = trace.values[:50] if hasattr(trace, 'values') else []
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
                                data_summary += f"{attr}: {attr_data[:50]}\n"
                
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
                sample_size = min(50, row_count)
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
                if len(counts) <= 50:  # Show all if not too many categories
                    desc = f"Series({len(result)}) value counts:\n{counts.to_string()}"
                else:
                    desc = f"Series({len(result)}) top value counts:\n{counts.head(50).to_string()}"
            else:
                # Show up to 50 items for any series
                max_display = min(50, len(result))
                desc = f"Series({len(result)}):\n{result.head(max_display).to_string()}"
                if len(result) > max_display:
                    desc += f"\n... and {len(result)-max_display} more"
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
    
    thinking_content, cleaned, ruminated, token_counts = call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=True,
        max_tokens=8192,
        thinking_placeholder=thinking_placeholder,
        thinking_title="Result Analysis Thinking"
    )
    
    logger.info(f"Completed streaming response, thinking content length: {len(thinking_content)}")
    logger.debug(f"Model thinking: {thinking_content}")
    logger.info(f"Reasoning token usage: {token_counts}")
    
    # Handle rumination case
    if ruminated:
        logger.warning("Rumination detected in reasoning agent, returning truncated explanation")
        cleaned = "The data analysis shows interesting patterns in the results. [Note: Analysis was truncated due to extended processing]"
    
    return thinking_content, cleaned, token_counts

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

    result, tokencounts = call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=False,
        max_tokens=8192
    )
    return result

# === Main Streamlit App ===============================================

def main():
    logger.info("Starting Data Analysis Agent application")
    st.set_page_config(layout="wide")
    if "reasoning_enabled" not in st.session_state:
        st.session_state.reasoning_enabled = False  # Default to disabled
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

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
    details.data {
        background-color: #f0f8ff;
        padding: 12px;
        border-radius: 8px;
        margin-top: 15px;
        border-left: 5px solid #3cb371;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    details.data summary {
        cursor: pointer;
        font-weight: bold;
        color: #333;
        margin-bottom: 8px;
    }
    details.data pre {
        background-color: #ffffff;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #e6e6e6;
        max-height: 300px;
        overflow-y: auto;
    }
    .thinking-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        background-color: #f9f9f9;
    }
    .token-counter {
        padding: 8px 12px;
        background-color: #f2f7ff;
        border-radius: 6px;
        border: 1px solid #c7d9f2;
        font-size: 0.9em;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([3,7])

    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by Azure AI Foundry NVIDIA Llama-3.3-Nemotron-Super-49</a></medium>", unsafe_allow_html=True)
        
        # Add the toggle switch
        reasoning_enabled = st.toggle("Enable Detailed Reasoning", value=st.session_state.reasoning_enabled)
        st.session_state.reasoning_enabled = reasoning_enabled
        
        # Display total tokens used across the session
        st.markdown(
            f"""<div class="token-counter">
            <span>Session tokens: <b>{st.session_state.total_tokens_used["total_tokens"]}</b></span>
            <span>Input: {st.session_state.total_tokens_used["input_tokens"]}</span>
            <span>Output: {st.session_state.total_tokens_used["output_tokens"]}</span>
            </div>""", 
            unsafe_allow_html=True
        )
        
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                logger.info(f"Loading new file: {file.name}")
                st.session_state.df = pd.read_csv(file)
                st.session_state.current_file = file.name
                st.session_state.messages = []
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = DataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df)
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
                    if msg.get("figure") is not None:
                        # Display Plotly figure directly
                        st.plotly_chart(msg["figure"], use_container_width=True)
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("code") is not None:
                        # Use Streamlit's native code display
                        with st.expander("View code"):
                            st.code(msg["code"], language="python")
                    if msg.get("token_counts") is not None:
                        st.markdown(
                            f"""<div class="token-counter">
                            <span>Tokens: <b>{msg["token_counts"]["total_tokens"]}</b></span>
                            <span>Input: {msg["token_counts"]["input_tokens"]}</span>
                            <span>Output: {msg["token_counts"]["output_tokens"]}</span>
                            </div>""", 
                            unsafe_allow_html=True
                        )

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about your data…"):
                logger.info(f"Received user query: {user_q}")
                st.session_state.messages.append({"role": "user", "content": user_q})
                
                # Display the user message immediately 
                with st.chat_message("user"):
                    st.markdown(user_q)
                
                # Create containers for thinking output that won't be overwritten
                thinking_container = st.container()
                with thinking_container:
                    st.markdown("#### Model Thinking Process")
                    code_thinking_placeholder = st.empty()
                    st.markdown("---")  # Separator between thinking blocks
                    reasoning_thinking_placeholder = st.empty()
                
                with st.spinner("Working …"):
                    # Pass the code thinking placeholder to CodeGenerationAgent
                    code, should_plot_flag, error_msg, thinking_content, code_token_counts = CodeGenerationAgent(
                        user_q, 
                        st.session_state.df,
                        thinking_placeholder=code_thinking_placeholder
                    )
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    
                    # Pass the reasoning thinking placeholder to ReasoningAgent
                    raw_thinking, reasoning_txt, reasoning_token_counts = ReasoningAgent(
                        user_q, 
                        result_obj, 
                        code,
                        thinking_placeholder=reasoning_thinking_placeholder
                    )
                    reasoning_txt = reasoning_txt.replace("`", "")
                    
                    # Combine token counts from code generation and reasoning
                    total_token_counts = {
                        "input_tokens": code_token_counts["input_tokens"] + reasoning_token_counts["input_tokens"],
                        "output_tokens": code_token_counts["output_tokens"] + reasoning_token_counts["output_tokens"],
                        "total_tokens": code_token_counts["total_tokens"] + reasoning_token_counts["total_tokens"]
                    }
                    
                    # Update session token totals
                    st.session_state.total_tokens_used["input_tokens"] += total_token_counts["input_tokens"]
                    st.session_state.total_tokens_used["output_tokens"] += total_token_counts["output_tokens"]
                    st.session_state.total_tokens_used["total_tokens"] += total_token_counts["total_tokens"]

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
                        '<hr/>'
                        '<strong>Generated Code:</strong>'
                        f'<pre><code class="language-python">{code}</code></pre>'
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

                # Create HTML for DataFrame/Series results if applicable
                data_result_html = ""
                if isinstance(result_obj, (pd.DataFrame, pd.Series, pd.Index)) and not isinstance(result_obj, str):
                    if isinstance(result_obj, pd.DataFrame):
                        # For DataFrames, format with to_html
                        result_display = result_obj.to_html(max_rows=20, classes="dataframe table table-striped")
                    elif isinstance(result_obj, pd.Index):
                        # For Index objects, convert to Series first then to string
                        result_display = pd.Series(result_obj).to_string()
                    else:
                        # For Series, convert to string representation
                        result_display = result_obj.to_string()
                    data_result_html = (
                        '<details class="data" open>'
                        '<summary>📊 Data Result</summary>'
                        f'<div style="max-height: 300px; overflow-y: auto;">{result_display}</div>'
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
                
                # Put together the final message with proper ordering:
                # 1. Thinking sections (if enabled)
                # 2. Plot (if any)
                # 3. Data result (in collapsible section)
                # 4. Explanation
                # 5. Code
                assistant_msg = f"{thinking_html}"
                # Note: Plot will be added separately via st.plotly_chart
                
                # Add data result HTML after plot (will appear before explanation)
                assistant_msg += data_result_html
                
                # Add explanation and code
                assistant_msg += f"{explanation_html}"

                logger.debug("Adding assistant response to session state")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "figure": figure,
                    "token_counts": total_token_counts,
                    "code": code  # Store the code separately for rendering with st.code()
                })
                st.rerun()

if __name__ == "__main__":
    main() 