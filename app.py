import os
import io
import json
import base64
import tempfile
import asyncio
import logging
import re
import sys
from typing import List, Dict, Any
import subprocess
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import requests
from mangum import Mangum
# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================

# --- Load Environment Variables ---
load_dotenv()

# --- API Keys and Model Configuration ---
OPENAI_URL=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 1))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_DEFAULT = "gpt-5"

# --- Logging Configuration ---
# Read log level from environment variable, default INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger()
logger.setLevel(getattr(logging, log_level, logging.INFO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent API",
    description="An agent that uses LLMs and a secure environment to analyze data.",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Clients ---
if not OPENAI_API_KEY:
    logging.warning("⚠️ OPENAI_API_KEY is not set. The application may not function correctly.")
    openai_client = None
else:
    logging.info(f'"Initializing OpenAI client...{OPENAI_URL}"')
    openai_client = OpenAI(base_url=OPENAI_URL,api_key=OPENAI_API_KEY)
    logging.info("✅ OpenAI client initialized.")


# ==============================================================================
# 2. INTEGRATED WEB DASHBOARD
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Serves a simple HTML dashboard for interacting with the API.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Data Analyst Agent</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
            .loader {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4f46e5;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body class="bg-gray-100 text-gray-800">
        <div class="container mx-auto p-4 md:p-8">
            <div class="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
                <h1 class="text-3xl font-bold mb-2 text-center text-gray-700">🤖 Data Analyst Agent</h1>
                <p class="text-center text-gray-500 mb-6">Submit your data analysis task below.</p>

                <form id="analysis-form" class="space-y-6">
                    <div>
                        <label for="questions-file" class="block text-sm font-medium text-gray-700 mb-1">Questions File (questions.txt)</label>
                        <input type="file" id="questions-file" name="questions.txt" accept=".txt" required
                            class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100 cursor-pointer" />
                    </div>
                    <div>
                        <label for="data-files" class="block text-sm font-medium text-gray-700 mb-1">Data Files (optional)</label>
                        <input type="file" id="data-files" name="data-files" multiple
                            class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100 cursor-pointer" />
                    </div>
                    <div>
                        <button type="submit"
                            class="w-full bg-indigo-600 text-white font-bold py-3 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-300 flex items-center justify-center">
                            <svg id="button-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-2"><path d="M12 20V10M18 20V4M6 20V16"/></svg>
                            <span id="button-text">Analyze</span>
                        </button>
                    </div>
                </form>

                <div id="loading" class="hidden flex flex-col justify-center items-center mt-8 text-center">
                    <div class="loader"></div>
                    <p class="mt-4 text-gray-600">Analyzing, please wait...</p>
                    <p class="text-sm text-gray-500">(Complex queries can take up to 6 minutes)</p>
                </div>

                <div id="results" class="mt-8 hidden">
                    <h2 class="text-2xl font-bold mb-4 text-center">Results</h2>
                    <div class="bg-gray-50 p-4 rounded-md shadow-inner">
                        <pre id="json-output" class="whitespace-pre-wrap break-all text-sm font-mono"></pre>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('analysis-form');
            const button = form.querySelector('button[type="submit"]');
            const buttonText = document.getElementById('button-text');
            const buttonIcon = document.getElementById('button-icon');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const loadingDiv = document.getElementById('loading');
                const resultsDiv = document.getElementById('results');
                const jsonOutput = document.getElementById('json-output');

                // Start loading state
                loadingDiv.classList.remove('hidden');
                resultsDiv.classList.add('hidden');
                button.disabled = true;
                buttonText.textContent = 'Analyzing...';

                const formData = new FormData(form);

                try {
                    const response = await fetch('/api/', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    
                    if (response.ok) {
                        jsonOutput.textContent = JSON.stringify(data, null, 2);
                    } else {
                        // Handle structured errors from the API
                        const errorJson = {
                            error: data.detail || "An error occurred.",
                            status_code: response.status
                        };
                        jsonOutput.textContent = JSON.stringify(errorJson, null, 2);
                    }
                    resultsDiv.classList.remove('hidden');

                } catch (error) {
                    console.error('Error:', error);
                    const errorJson = {
                        error: "Failed to fetch results from the server.",
                        details: error.message
                    };
                    jsonOutput.textContent = JSON.stringify(errorJson, null, 2);
                    resultsDiv.classList.remove('hidden');
                } finally {
                    // End loading state
                    loadingDiv.classList.add('hidden');
                    button.disabled = false;
                    buttonText.textContent = 'Analyze';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ==============================================================================
# 3. CORE API LOGIC
# ==============================================================================

def openai_chat_code_gen(messages, model: str = OPENAI_MODEL_DEFAULT, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """Specialized wrapper for generating Python code from the OpenAI LLM."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Check API key.")
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return resp.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with LLM provider: {e}")

def extract_python_code(llm_response: str) -> str:
    """Extracts the Python code from a markdown code block."""
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback if the model doesn't use markdown but provides raw code
    if "import pandas" in llm_response or "import json" in llm_response:
        logging.warning("LLM response did not contain a markdown block. Falling back to raw content.")
        return llm_response.strip()
    raise ValueError("Could not extract Python code from the LLM response.")


class CaptureFD:
    """Capture both sys.stdout/stderr and low-level fd writes."""
    def __enter__(self):
        import tempfile
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        self._stdout_tmp = tempfile.TemporaryFile(mode="w+b")
        self._stderr_tmp = tempfile.TemporaryFile(mode="w+b")

        os.dup2(self._stdout_tmp.fileno(), 1)
        os.dup2(self._stderr_tmp.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)

        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)

        self._stdout_tmp.seek(0)
        self._stderr_tmp.seek(0)
        self.stdout = self._stdout_tmp.read().decode(errors="ignore")
        self.stderr = self._stderr_tmp.read().decode(errors="ignore")

        self._stdout_tmp.close()
        self._stderr_tmp.close()

def execute_generated_code(script: str, files: Dict[str, bytes]) -> Dict[str, Any]:
    """Executes Python code, capturing stdout/stderr reliably."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files.items():
            if filename:
                with open(os.path.join(temp_dir, filename), "wb") as f:
                    f.write(content)

        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            logging.info(f"Running generated code in-process within {temp_dir}")

            with CaptureFD() as cap:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    try:
                        exec(script, {"__name__": "__main__"})
                    except SystemExit as e:
                        logging.warning(f"Script called sys.exit({e.code})")
                        return {
                            "stdout": cap.stdout + buf_out.getvalue(),
                            "stderr": cap.stderr + buf_err.getvalue(),
                            "exit_code": e.code
                        }

            stdout = cap.stdout + buf_out.getvalue()
            stderr = cap.stderr + buf_err.getvalue()
            return {"stdout": stdout, "stderr": stderr, "exit_code": 0}

        except Exception as e:
            error_details = f"{type(e).__name__}: {str(e)}"
            logging.error(f"Execution error: {error_details}")
            return {
                "stdout": "",
                "stderr": f"{error_details}\n{e}",
                "exit_code": 1
            }
        finally:
            os.chdir(original_cwd)

@app.post("/api")
@app.post("/api/")
async def api(request: Request):
    """
    Main API endpoint that receives multipart form data and processes the analysis request.
    """
    logging.info("Received a new analysis request.")
    # Increased timeout for the entire request
    timeout_sec = int(os.getenv("AGENT_TIMEOUT_SEC", "350"))
    try:
        return await asyncio.wait_for(handle_request(request), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logging.error("Request timed out after %s seconds.", timeout_sec)
        raise HTTPException(status_code=504, detail=f"Request timed out after {timeout_sec} seconds")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Top-level error handler caught: {e}", exc_info=True)
        detail = str(e)
        return JSONResponse(
            content={"error": "An unexpected server error occurred.", "detail": detail}, 
            status_code=500
        )


async def handle_request(request: Request):
    """
    Parses the multipart form request, runs the analysis, and returns the result.
    """
    form = await request.form()
    

    logging.info("Parsing form data from the request.")
    logging.info(f"Form data keys: {list(form.keys())}")

    qfile: UploadFile = form.get("questions.txt")
    if not qfile:
        txt_files = [item for item in form.values() if isinstance(item, UploadFile) and item.filename.lower().endswith(".txt")]
        if len(txt_files) == 1: 
            qfile = txt_files[0]
            logging.warning(f"Could not find 'questions.txt', but found '{qfile.filename}'. Using it as the questions file.")
        else: 
            raise HTTPException(status_code=400, detail="A single 'questions.txt' file is required.")

    # Correctly handle multiple optional file uploads and filter out empty ones
    attachment_key =[f for f in form.keys()]
    attachments = []
    for key in attachment_key:
        attachments = [f for f in form.getlist(key) if f.filename and f.filename.lower() != "questions.txt"]
    
    question_text = (await qfile.read()).decode('utf-8')
    attachment_files = {f.filename: await f.read() for f in attachments}
    
    # Intelligent URL scraping: only scrape if the query explicitly asks for it.
    html_content = ""
    if "scrape" in question_text.lower() or "from the url" in question_text.lower():
        url_match = re.search(r"https?://\S+", question_text)
        if url_match:
            url = url_match.group(0).rstrip('.,)!?]>') # Clean trailing punctuation
            logging.info(f"Query requests scraping. Found and cleaned URL: {url}")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                html_content = response.text
                logging.info(f"Successfully scraped {len(html_content)} characters from {url}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to fetch URL {url}: {e}")
                # Provide a meaningful error in the HTML content for the LLM to see.
                html_content = f"<html><body><p>Failed to fetch URL: {e}</p></body></html>"

    all_input_files = {"questions.txt": question_text.encode('utf-8'), **attachment_files}
    if html_content:
        all_input_files["scraped_page.html"] = html_content.encode('utf-8')
    
    logging.info(f"Files being passed to execution environment: {list(all_input_files.keys())}")



    system_prompt = (
        "You are an expert Python data analyst. Your sole task is to generate a complete, self-contained, and robust Python script to answer the user's question. "
        "The script will be executed in a secure environment with pandas, matplotlib, beautifulsoup4, and duckdb installed. "
        "The script's **ONLY** output to **standard output (stdout)** must be a **single JSON array or object** that contains the final, raw answer values. Do not include descriptive text. All other logs or debug information must be written to **standard error (stderr)**.\n\n"
        "Respond ONLY with the Python code inside a markdown block: ```python\n...code...\n```\n\n"
        "---"
        "CRITICAL REQUIREMENTS FOR THE GENERATED SCRIPT:\n"
        "1.  **Imports**: Always start the script with all necessary imports, including `import pandas as pd`, `import json`, `import numpy as np`, `import sys`, `import duckdb`, and `import matplotlib`. **Crucially, you must set the Matplotlib backend to 'Agg' immediately after importing it: `matplotlib.use('Agg')` to prevent GUI errors in a server environment.**\n"
        "2.  **Debugging**: For debugging, the very first thing your script should do is print the list of files in the current directory to stderr. Example: `import os, sys; print(f'Files in current directory: {os.listdir(\".\")}', file=sys.stderr)`\n"
        "   a. Except for the final result, all other `print()` statements must direct output to `sys.stderr`. No intermediate or debug prints to stdout are allowed.\n"
        "3.  **Determine Data Source & Task Type**: After debugging, analyze the user's question and the list of available files to determine the task."
        "    - **Scenario A: Web Scraping.** If the user asks to 'scrape a URL' and `scraped_page.html` is in the 'Files available' list, your primary data source is that HTML file. "
        "    - **Scenario B: File Analysis.** If the user asks to analyze a specific file (e.g., 'Analyze `sample-sales.csv`') AND that file's name appears in the 'Files available' list, you MUST load it directly by its name from the current directory (e.g., `pd.read_csv('sample-sales.csv')`). "
        "    - **Scenario C: Remote Data Query.** If the user's query *describes* a remote dataset and provides a **SQL query** (especially a DuckDB query for S3), your script **must** execute this query using DuckDB to fetch the data into a pandas DataFrame, and then perform the analysis. The necessary DuckDB extensions (httpfs, parquet) are installed. "
        "    - **Error Case:** If the script determines it's a File Analysis task (Scenario B) but the required file is missing from the file list printed in the debug step, *then* it should exit with a JSON error: `{\"error\": \"The required data file was not provided. Please upload the file and try again.\"}`. "
        "    - **Do not make any external HTTP requests.**\n"
        "4.  **Error Handling**: The script must be resilient. Use `try-except` blocks for all major operations. If an unrecoverable error occurs, print a JSON object like `{\"error\": \"Descriptive error message\"}` and exit immediately.\n"
        "5.  **Intelligent HTML Table Parsing**: If using `scraped_page.html`, infer keywords from the user's question to find the correct `<table>`. If no suitable table is found, exit with a specific JSON error.\n"
        "6.  **MANDATORY Data Cleaning & Column Mapping**:\n"
        r"    a. **Clean Column Names**: After loading data, robustly clean column names. First, ensure all are strings: `df.columns = [str(c) for c in df.columns]`. Then, apply cleaning: `df.columns = df.columns.str.lower().str.strip().str.replace(r'\[.*?\]', '', regex=True).str.replace(r'[^\w]+', '_', regex=True)`." + "\n"
        r"    b. **Map Concepts to Cleaned Columns**: After cleaning, map concepts from the user's question (e.g., 'movie name') to the *exact* cleaned column names (e.g., `df['title']`). **Do not use assumed or hallucinated names.**" + "\n"
        r"    c. **Clean Cell Values for Numerics**: Before numeric operations, clean the relevant columns using this exact three-step process:" + "\n"
        r"        i. **Force to String**: `df['col'] = df['col'].astype(str)`." + "\n"
        r"        ii. **Remove Non-Numeric Chars**: `df['col'] = df['col'].str.replace(r'[^\d.]', '', regex=True)`." + "\n"
        r"        iii. **Convert to Numeric**: `df['col'] = pd.to_numeric(df['col'], errors='coerce')`." + "\n"
        r"    d. **Verify Required Columns Exist**: Before accessing any column (e.g., `df['precipitation_mm']`), first check whether it exists in the cleaned `df.columns`. "
        r"      If not, return a JSON error like: `{\"error\": \"Required column 'precipitation_mm' not found.\", \"available_columns\": list(df.columns)}` and exit.\n"
        "7.  **Handle NaN Values for JSON**: Before creating the final JSON, convert any `NaN` values from calculations to `None` to ensure correct serialization to `null`. Use `pd.isna()` for checks.\n"
        "8.  **Plotting**: If a plot is requested, you **MUST** first handle potential `NaN` values in the plotting columns by dropping rows with missing values in those specific columns (e.g., `plot_df = df.dropna(subset=['x_col', 'y_col'])`). Use this `plot_df` for plotting. Save the plot to `plot.png`, encode it as base64, and include it in the final JSON as a data URI.\n"
        "     a. By default, save plots as `plot.png`. If the user specifies a different format (like SVG), use that format and update the data URI prefix accordingly (`data:image/svg+xml;base64,...`).\n"
        "9.  **Final Output**: Before printing the final JSON, you **MUST** ensure all data within your final list or dictionary is JSON serializable. This is especially important for numbers from pandas/numpy. Define and use a helper function to recursively convert any `np.int64`, `np.float64`, etc., to native Python `int` or `float` types. The script's final action must be `print(json.dumps(final_answer_dict_or_list),file=sys.stdout)`. The output must be a JSON object or list as requested. Example: `[1, \"Titanic\", 2.26, \"data:image/png;base64,...\"]` or `{\"total_sales\": 1000, ...}`.\n"
        "     9a. If generating plots, wrap the entire plotting code in a try-except block. If an error occurs while plotting, omit the plot from the final JSON but continue returning the result.\n"
        "     9b. The script must define and use a `def convert_types(obj):` function that recursively converts NumPy/Pandas types to native Python types. Use this to clean the final result before dumping it with `json.dumps(...)`.\n"
        "10. You must not call `exit()` or `sys.exit()` in the script. Any function that may trigger these (e.g., via a library) should be used inside a try/except block to catch `SystemExit` and convert it to a JSON error."
    )

    system_prompt = os.getenv("SYSTEM_PROMPT", system_prompt)
    
    user_prompt = (
        f"User Question:\n---\n{question_text}\n---\n\n"
        f"Files available in the current directory: {', '.join(all_input_files.keys())}\n\n"
        "Please generate the complete Python script now."
    )

    logging.info(f"user prompt: {user_prompt}")

    iteration = 0
    llm_response_text=""
    stdout=""
    message= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
    while iteration < RETRY_ATTEMPTS:
        if iteration>0:
            logging.info(f"Iteration {iteration}: Regenerating code due to previous error.")
            additionalMessage = [
                {"role": "assistant", "content": f"{extract_python_code(llm_response_text)}"},
                {"role": "user", "content": f"That python code doesn't provide any result. Captured stdout:{stdout}. Can you regenerate python script answer to that question?"}
            ]
            logging.info(f"Addtional message: {additionalMessage}")

            message.extend(additionalMessage)

        logging.info(f"Iteration {iteration}: Sending request to LLM for code generation.")
        logging.info(f"Current message content: {message}")
        llm_response_text = openai_chat_code_gen(
            messages=message
        )
        
        logging.info(f"LLM response received. Length: {len(llm_response_text)} characters.")
        logging.info(f"LLM response content: {llm_response_text}")
        try:
            python_code = extract_python_code(llm_response_text)
            logging.info(f"Extracted Python code from LLM response: {python_code}")
        except ValueError as e:
            logging.error(f"Failed to extract code from LLM response: {llm_response_text}")
            raise HTTPException(status_code=500, detail=f"Could not generate a valid script from the LLM. Response: {llm_response_text}")

        exec_result = await asyncio.to_thread(execute_generated_code, python_code, all_input_files)

        stdout = exec_result.get("stdout", "").strip()
        stderr = exec_result.get("stderr", "").strip()
        
        logging.info(f"Execution completed. Exit code: {exec_result['exit_code']}")
        logging.info(f"Captured stdout: {stdout}")
        logging.info(f"Captured stderr: {stderr}")
        try:
            final_json_output = json.loads(stdout)
            if isinstance(final_json_output, dict) and 'error' in final_json_output:
                logging.error(f"Script executed but returned a controlled error: {final_json_output['error']}")
                raise HTTPException(status_code=422, detail=final_json_output['error'])
            
            return JSONResponse(content=final_json_output)
        
        except json.JSONDecodeError:
            if exec_result["exit_code"] != 0:
                error_message = f"Script execution failed with exit code {exec_result['exit_code']}."
                details = stderr if stderr else stdout
                logging.error(f"{error_message} Details: {details}")
                if iteration+1 == RETRY_ATTEMPTS:
                    raise HTTPException(status_code=500, detail=f"{error_message} Details: {details}")
            else:
                error_message = "Script executed successfully but produced non-JSON output."
                logging.error(f"{error_message} Output: {stdout}")
                if iteration+1 == RETRY_ATTEMPTS:
                    raise HTTPException(status_code=500, detail=f"{error_message} Output: {stdout}")
            iteration+= 1
        


# ==============================================================================
# 4. APPLICATION RUNNER
# ==============================================================================

handler = Mangum(app)

# if __name__ == "__main__" or "AWS_LAMBDA_FUNCTION_NAME" not in os.environ:
#     import uvicorn
#     # if not os.path.exists("Dockerfile"):
#     #     logging.warning("⚠️ Dockerfile not found. Make sure you have a Dockerfile to build the agent's execution environment.")
#     print("Starting server in development mode...")
#     port = int(os.getenv("PORT", "10001"))
#     logging.info(f"🚀 Starting server on http://0.0.0.0:{port}")
#     uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
