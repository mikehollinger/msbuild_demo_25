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
import matplotlib.pyplot as plt
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
formatter = ColoredLevelFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
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
    max_tokens: int = 1024, 
    thinking_placeholder: Optional[Any] = None,
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
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
        
    Returns:
        If stream=False: Just the response content
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
            return result
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
                                f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
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

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    system_content = "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."
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
    intent_response = result.strip().lower()
    logger.info(f"Query understanding result: {intent_response} for query: {query}")
    return intent_response == "true"

# === CodeGeneration TOOLS ============================================

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    logger.debug(f"Generating plot code prompt for query: {query}")
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    3. Create only ONE relevant plot. Set `figsize=(6,4)`, add title/labels.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
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
    """

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame, max_retries: int = 3):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    logger.info(f"CodeGenerationAgent processing query: {query}")
    should_plot = QueryUnderstandingTool(query)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)
    
    code = ""
    error_msg = ""
    retries = 0
    
    system_content = "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."
    
    while retries <= max_retries:
        # If this is a retry, include the error message in the prompt
        retry_prompt = prompt
        if retries > 0:
            retry_prompt = f"""
            {prompt}
            
            The previous code generated an error:
            {error_msg}
            
            Please fix the code to avoid this error.
            """
            logger.info(f"Retrying code generation (attempt {retries}/{max_retries}) after error: {error_msg}")
        
        result = call_llm_api(
            prompt=retry_prompt,
            system_content=system_content,
            stream=False,
            temperature=0.2,
            max_tokens=1024
        )

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
                return code, should_plot, error_msg
        
        retries += 1
    
    # If we've reached max retries and still have errors, return the last code anyway
    logger.warning(f"Reached maximum retries ({max_retries}) with errors, returning last generated code")
    return code, should_plot, error_msg

# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    logger.debug(f"Executing generated code:\n{code}")
    
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
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
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    logger.debug("Building reasoning prompt")
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        # Extract title and numerical data summary from the plot
        title = ""
        data_summary = ""
        
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
            # Extract data from the axes
            axes = result.get_axes()
            if axes:
                # Get data from first axis
                ax = axes[0]
                
                # Get axis labels
                x_label = ax.get_xlabel()
                y_label = ax.get_ylabel()
                
                # Try to extract data with category labels
                if hasattr(ax, 'containers') and ax.containers:
                    # For bar charts
                    container_data = []
                    if ax.containers and len(ax.containers) > 0:
                        # Get x-ticks which often correspond to categories
                        x_ticks = ax.get_xticks()
                        tick_labels = [str(label) for label in ax.get_xticklabels()]
                        
                        # Extract text representations from tick labels
                        if tick_labels and all(label.get_text() for label in ax.get_xticklabels()):
                            tick_labels = [label.get_text() for label in ax.get_xticklabels()]
                        
                        # Match bars with their categories
                        if len(tick_labels) > 0 and len(tick_labels) == len(x_ticks):
                            bar_heights = [p.get_height() for p in ax.patches] if hasattr(ax, 'patches') and ax.patches else []
                            
                            # Create category-value pairs
                            if bar_heights and len(bar_heights) == len(tick_labels):
                                pairs = [f"{label}: {height}" for label, height in zip(tick_labels, bar_heights)]
                                data_summary = f"Data: {', '.join(pairs)}"
                            else:
                                # Fallback if exact matching failed
                                container = ax.containers[0]
                                if hasattr(container, 'datavalues'):
                                    values = container.datavalues
                                    if len(values) <= len(tick_labels):
                                        pairs = [f"{label}: {val}" for label, val in zip(tick_labels[:len(values)], values)]
                                        data_summary = f"Data: {', '.join(pairs)}"
                
                # Try to get x and y data for line plots
                if not data_summary and hasattr(ax, 'lines') and ax.lines:
                    line = ax.lines[0]
                    x_data = line.get_xdata()[:5]
                    y_data = line.get_ydata()[:5]
                    
                    # Try to get legend labels
                    legend_labels = []
                    if ax.get_legend():
                        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
                    
                    if legend_labels:
                        data_summary = f"Series: {legend_labels[0] if legend_labels else ''}, Points: {list(zip(x_data, y_data))}"
                    else:
                        data_summary = f"X values: {x_data}, Y values: {y_data}"
                
                if x_label or y_label:
                    data_summary += f"\nAxes: {x_label or 'x'} vs {y_label or 'y'}"
        
        elif isinstance(result, plt.Axes):
            title = result.get_title()
            # Apply similar extraction logic here
            # Simplified for brevity
            
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
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows and its key insights.'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result is:
        {desc}
        Explain in 2–3 concise sentences what this tells about the data.'''
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    logger.info("Generating reasoning about results")
    prompt = ReasoningCurator(query, result)
    logger.info(f"Reasoning prompt: {prompt}")

    # Get the current reasoning state
    reasoning_enabled = st.session_state.get("reasoning_enabled", True)
    system_content = "detailed thinking on. You are an insightful data analyst." if reasoning_enabled else "detailed thinking off. You are an insightful data analyst."

    # Use the unified API calling function with streaming
    thinking_placeholder = st.empty()
    
    thinking_content, cleaned = call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=True,
        temperature=0.2,
        max_tokens=1024,
        thinking_placeholder=thinking_placeholder
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
    
    system_content = "detailed thinking off. You are a data analyst providing brief, focused insights."
    
    return call_llm_api(
        prompt=prompt,
        system_content=system_content,
        stream=False,
        temperature=0.2,
        max_tokens=512
    )

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Main Streamlit App ===============================================

def main():
    logger.info("Starting Data Analysis Agent application")
    st.set_page_config(layout="wide")
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "reasoning_enabled" not in st.session_state:
        st.session_state.reasoning_enabled = True  # Default to enabled

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
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Display plot at fixed size
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about your data…"):
                logger.info(f"Received user query: {user_q}")
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Working …"):
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build assistant response
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    logger.debug("Storing generated plot")
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                else:
                    header = f"Result: {result_obj}"

                # Show only reasoning thinking in Model Thinking (collapsed by default)
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Add retry information if there were retries
                if code_thinking:
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
                # Combine thinking, explanation, and code accordion
                assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                logger.debug("Adding assistant response to session state")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()

if __name__ == "__main__":
    main() 