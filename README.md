# 🚀 ASAVE (AAOIFI Standard Augmentation & Validation Engine) - API

ASAVE API provides endpoints to leverage AI for reviewing, suggesting, and validating updates to AAOIFI Financial Accounting Standards (FAS) and Shari'ah Standards (SS). It allows for granular analysis of text chunks with contextual information and provides detailed, step-by-step responses from its specialized AI agents.

## 📚 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Prerequisites](#3-prerequisites)
4. [Setup & Installation (Windows)](#4-setup--installation-windows)
5. [Running the API Server (Windows)](#5-running-the-api-server-windows)
6. [API Endpoints](#6-api-endpoints)
    * [GET /status](#get-status)
    * [POST /initialize](#post-initialize)
    * [POST /analyze_chunk](#post-analyze_chunk)
    * [POST /mine_shariah_rules](#post-mine_shariah_rules)
7. [Example API Calls (Windows - PowerShell/CMD and Postman)](#7-example-api-calls-windows---powershellcmd-and-postman)
    * [Initializing the System](#initializing-the-system)
    * [Analyzing a Text Chunk](#analyzing-a-text-chunk)
    * [Mining Shari'ah Rules](#mining-shariah-rules)
    * [Checking System Status](#checking-system-status)
8. [Project Structure](#8-project-structure)
9. [Important Considerations](#9-important-considerations)

---

## 1️⃣ Project Overview

The ASAVE API is built with Flask and utilizes a multi-agent system powered by Google Gemini Large Language Models via Langchain. It processes AAOIFI standards (PDFs), a curated Shari'ah knowledge base, and user-provided text to:

- 💡 Generate suggestions for clarifying or enhancing AAOIFI FAS.
- ✅ Validate these suggestions for Shari'ah compliance against explicit rules and Shari'ah Standards.
- 🔄 (Conceptually) Check for inter-standard consistency.
- 🏷️ Mine Shari'ah rules from AAOIFI Shari'ah Standard documents.

The API is designed to return detailed JSON responses that include the outputs of each agent involved in the processing pipeline, offering a "thinking process" view.

---

## 2️⃣ Features

- 📄 **Document Processing:** Loads and chunks PDF documents (FAS and SS).
- 🧠 **Vector Store Creation:** Generates embeddings and stores them in ChromaDB for efficient retrieval (RAG).
- 🤖 **AI-Powered Suggestion Generation (AISGA):** Proposes modifications or new clauses for FAS text based on context.
- 🕵️ **Shari'ah Compliance Validation (SCVA):** Validates suggestions against explicit rules and semantically against SS documents.
- 🔗 **Inter-Standard Consistency Check (ISCCA):** (Conceptual) Checks suggestions for consistency with other FAS.
- 🏷️ **Shari'ah Rule Mining (SRMA):** Extracts and structures Shari'ah rules from SS documents.
- 📝 **Detailed API Responses:** JSON outputs detailing each agent's contribution and "thinking process".
- 📁 **File-based Initialization:** System is initialized using uploaded PDF and JSON files.

---

## 3️⃣ Prerequisites

- 🐍 Python 3.9+ (add to Windows PATH)
- 📦 `pip` (usually comes with Python)
- 🔑 **Google Gemini API Key:** Obtain from Google AI Studio or Google Cloud.
- 📄 AAOIFI FAS and SS documents in PDF format.
- 🗂️ (Optional) `shariah_rules_explicit.json` file with pre-defined Shari'ah rules.
- 🧪 Postman (for easy API testing)
- 💻 `curl` for Windows (via Git Bash or install separately)

---

## 4️⃣ Setup & Installation (Windows)

1. **Clone the Repository (or create the project structure):**
    ```bash
    git clone <your-repo-url>
    cd asave_api_project
    ```
    Or manually create the directory structure as in [Project Structure](#8-project-structure).

2. **Set Google API Key (Windows):**
    - **Command Prompt:**
        ```cmd
        set GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
        ```
    - **PowerShell:**
        ```powershell
        $env:GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        ```
    - **Permanently:**
        1. Search for "environment variables"
        2. Edit system environment variables
        3. Add `GOOGLE_API_KEY` with your key

3. **Create a Virtual Environment (Recommended):**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

4. **Install Dependencies:**
    ```cmd
    pip install -r requirements.txt
    ```
    Example `requirements.txt`:
    ```txt
    Flask>=2.0
    werkzeug>=2.0
    langchain>=0.1.0
    langchain-community>=0.0.15
    langchain-google-genai>=0.1.0
    google-generativeai>=0.4.0
    pymupdf>=1.23.0
    chromadb>=0.4.0
    tiktoken
    # reportlab # Only if running specific dummy PDF generation tests
    ```

5. **Prepare Input Files:**
    - Place AAOIFI FAS and SS PDFs in an accessible folder (e.g., `C:\asave_test_files\`)
    - Place or create `shariah_rules_explicit.json` in the project root

---

## 5️⃣ Running the API Server (Windows)

Once setup is complete, run the Flask API server from your activated virtual environment:

```cmd
python api_server.py
```

- 🌐 By default, server runs at [http://localhost:5001](http://localhost:5001)
- 🗂️ The server creates:
    - `temp_api_uploads\` (temporary uploads)
    - `db_store_api\` (ChromaDB vector stores)
    - `output_srma_api\` (SRMA results)

---

## 6️⃣ API Endpoints

- **GET /status**  
  _Returns current operational status_
- **POST /initialize**  
  _Initializes the ASAVE system_
- **POST /analyze_chunk**  
  _Analyzes a specific text chunk_
- **POST /mine_shariah_rules**  
  _Triggers SRMA_

See detailed request/response structures in the full documentation.

---

## 7️⃣ Example API Calls (Windows - PowerShell/CMD and Postman)

### 🟢 Initializing the System

**Postman:**  
- Import the collection, set `{{baseUrl}}` to `http://localhost:5001`
- Use "Select Files" for file parameters

**curl (CMD):**
```cmd
curl -X POST ^
  -F "fas_files=@C:/asave_test_files/dummy_fas.pdf" ^
  -F "ss_files=@C:/asave_test_files/dummy_ss.pdf" ^
  -F "shariah_rules_explicit_file=@C:/asave_test_files/shariah_rules_explicit.json" ^
  -F "persist_db_path_base=./curl_test_db_win" ^
  http://localhost:5001/initialize
```
> `^` is the line continuation for CMD. Use `\` for Git Bash.

---

### 🟢 Analyzing a Text Chunk

**Postman:**  
- Open "Analyze Text Chunk", edit JSON body, click "Send"

**curl (CMD):**
1. Save JSON to `payload_analyze.json`:
    ```json
    {
        "target_text_chunk": "An entity shall recognise revenue when (or as) the entity satisfies a performance obligation by transferring a promised good or service (i.e., an asset) to a customer.",
        "fas_context_chunks": [
            "Revenue is income arising in the course of an entity’s ordinary activities.",
            "A performance obligation is a promise in a contract with a customer to transfer to the customer either a good or service (or a bundle of goods or services) that is distinct."
        ],
        "ss_context_chunks": [
            "Contracts of exchange (Mu’awadat) must be free from major uncertainty (Gharar) and gambling (Maysir).",
            "The subject matter of a sale must be in existence, owned by the seller, and capable of delivery."
        ],
        "fas_name_for_display": "FAS on Revenue Recognition (Windows Test)",
        "identified_ambiguity": "The term asset transfer might need clarification on constructive transfer or control for Shariah compliance in specific Islamic finance contracts."
    }
    ```
2. Run:
    ```cmd
    curl -X POST -H "Content-Type: application/json" -d "@payload_analyze.json" http://localhost:5001/analyze_chunk
    ```

---

### 🟢 Mining Shari'ah Rules

**Postman:**  
- Use "Mine Shari'ah Rules (SRMA)", select files, fill form fields

**curl (CMD):**
```cmd
curl -X POST ^
  -F "ss_files_for_srma=@C:/asave_test_files/dummy_ss_for_srma.pdf" ^
  -F "ss_files_for_srma_0_fullname=Dummy Shariah Standard for SRMA (Win)" ^
  -F "ss_files_for_srma_0_shortcode=DWSRM1" ^
  -F "output_directory=./srma_curl_output_win" ^
  http://localhost:5001/mine_shariah_rules
```

---

### 🟢 Checking System Status

**Postman:**  
- Open "System Status", click "Send"

**curl:**
```cmd
curl http://localhost:5001/status
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:5001/status
```

---

## 8️⃣ Project Structure

```
asave_api_project/
├── api_server.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── ... (other agent files)
├── utils/
│   ├── __init__.py
│   └── document_processor.py
├── shariah_rules_explicit.json
├── temp_api_uploads/
├── db_store_api/
├── output_srma_api/
└── requirements.txt
```

---

## 9️⃣ Important Considerations

- 🔑 **API Key:** `GOOGLE_API_KEY` must be set for AI agents to function.
- 📁 **File Paths:** Use forward slashes (`C:/path/to/file`) or properly escaped backslashes (`C:\\path\\to\\file`) in Windows.
- ⚠️ **Error Handling, Security, State Management, Resource Intensity, Human Oversight:** Remain critical.
- 📝 **Logging:** Check console output of `python api_server.py` for logs.

---

### 📝 **Key Windows-specific changes:**

1. 🖥️ **Environment Variables:** Instructions for Command Prompt, PowerShell, and permanent settings.
2. 🐍 **Virtual Environment Activation:** Updated activation command.
3. 💻 **curl Examples:** Notes on path formats and line continuation (`^` for CMD). PowerShell users can use `Invoke-RestMethod`.
4. 📁 **File Path Guidance:** Reminders about Windows file path conventions.

> The Postman collection JSON remains unchanged; simply use the "Select Files" dialog to point to files on your Windows system.
