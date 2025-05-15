# 🚀 ASAVE (AAOIFI Standard Augmentation & Validation Engine) - API

ASAVE API provides endpoints to leverage AI for reviewing, suggesting, and validating updates to AAOIFI Financial Accounting Standards (FAS) and Shari'ah Standards (SS). It allows for granular analysis of text chunks with contextual information and provides detailed, step-by-step responses from its specialized AI agents.

---

## 📚 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Prerequisites](#3-prerequisites)
4. [Setup & Installation](#4-setup--installation)
5. [Running the API Server](#5-running-the-api-server)
6. [API Endpoints](#6-api-endpoints)
7. [Example API Calls (cURL)](#7-example-api-calls-curl)
8. [Project Structure](#8-project-structure)
9. [Important Considerations](#9-important-considerations)

---

## 1. 📝 Project Overview

The ASAVE API is built with Flask and utilizes a multi-agent system powered by Google Gemini Large Language Models via Langchain. It processes AAOIFI standards (PDFs), a curated Shari'ah knowledge base, and user-provided text to:

- 💡 Generate suggestions for clarifying or enhancing AAOIFI FAS.
- ✅ Validate these suggestions for Shari'ah compliance against explicit rules and Shari'ah Standards.
- 🔄 (Conceptually) Check for inter-standard consistency.
- 🏷️ Mine Shari'ah rules from AAOIFI Shari'ah Standard documents.

The API returns detailed JSON responses that include the outputs of each agent involved in the processing pipeline, offering a "thinking process" view.

---

## 2. ✨ Features

- 📄 **Document Processing:** Loads and chunks PDF documents (FAS and SS).
- 🧠 **Vector Store Creation:** Generates embeddings and stores them in ChromaDB for efficient retrieval (RAG).
- 🤖 **AI-Powered Suggestion Generation (AISGA):** Proposes modifications or new clauses for FAS text based on context.
- 🕵️ **Shari'ah Compliance Validation (SCVA):** Validates suggestions against explicit rules and semantically against SS documents.
- 🔗 **Inter-Standard Consistency Check (ISCCA):** (Conceptual) Checks suggestions for consistency with other FAS.
- 🏷️ **Shari'ah Rule Mining (SRMA):** Extracts and structures Shari'ah rules from SS documents.
- 📝 **Detailed API Responses:** JSON outputs detailing each agent's contribution and "thinking process".
- 📁 **File-based Initialization:** System is initialized using uploaded PDF and JSON files.

---

## 3. ⚙️ Prerequisites

- Python 3.9+
- `pip` and `venv` (recommended for virtual environments)
- **Google Gemini API Key:** You must have a valid API key from Google AI Studio or Google Cloud.
- AAOIFI FAS and SS documents in PDF format.
- (Optional) A `shariah_rules_explicit.json` file containing pre-defined Shari'ah rules.

---

## 4. 🛠️ Setup & Installation

1. **Clone the Repository (or create the project structure):**
    ```bash
    git clone <your-repo-url>
    cd asave_api_project
    ```
    Or create the directory structure as outlined in the [Project Structure](#8-project-structure) section.

2. **Set Google API Key:**
    ```bash
    export GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
    ```
    *Tip: For persistent setup, add this to your shell's profile file or use a `.env` file.*

3. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. **Install Dependencies:**
    ```bash
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
    - Place your AAOIFI FAS PDF files and SS PDF files in a location accessible for uploading via API calls.
    - Create or place your `shariah_rules_explicit.json` file in the project root if you have one.

---

## 5. 🚦 Running the API Server

Once setup is complete, run the Flask API server:

```bash
python api_server.py
```

- By default, the server will start on [http://localhost:5001](http://localhost:5001).
- The API server creates the following directories if they don't exist:
  - `temp_api_uploads/` : For temporarily storing files uploaded via API.
  - `db_store_api/` : Default location for persisted ChromaDB vector stores.
  - `output_srma_api/` : Default output directory for SRMA results.

---

## 6. 🔌 API Endpoints

All request bodies and responses are in JSON format, except for file uploads which use `multipart/form-data`.

### 🟢 `GET /status`

- **Description:** Returns the current operational status of the ASAVE API.
- **Response Example:**
  ```json
  {
     "service_status": "running",
     "asave_initialized": true,
     "config": {
        "google_api_key_set": true,
        "upload_folder": "temp_api_uploads",
        "explicit_rules_path": "shariah_rules_explicit.json",
        "mined_rules_path_default_location": "output_srma_api/shariah_rules_mined_combined.json"
     },
     "components_loaded": {
        "doc_processor": true,
        "fas_vector_store": true,
        "ss_vector_store": true,
        "aisga": true,
        "scva_iscca": true,
        "srma": true
     }
  }
  ```

### 🟠 `POST /initialize`

- **Description:** Initializes the ASAVE system. Must be called before using `/analyze_chunk` or `/mine_shariah_rules`.
- **Request:** `multipart/form-data` with FAS/SS PDFs and optional JSON.
- **Response Example:**
  ```json
  {
     "status": "success",
     "message": "ASAVE system initialized.",
     "fas_vector_store_status": "Created/Loaded",
     "ss_vector_store_status": "Created/Loaded",
     "explicit_shariah_rules_path": "path/to/used/shariah_rules_explicit.json"
  }
  ```

### 🟡 `POST /analyze_chunk`

- **Description:** Analyzes a specific text chunk from an FAS document.
- **Request:** JSON body with target chunk, context, and optional ambiguity.
- **Response Example:** (truncated for brevity)
  ```json
  {
     "status": "success",
     "analysis": {
        "input_summary": { ... },
        "aisga_step": { ... },
        "scva_step": { ... },
        "iscca_step": { ... }
     }
  }
  ```

### 🟣 `POST /mine_shariah_rules`

- **Description:** Triggers the Shari'ah Rule Miner Agent (SRMA) to process SS PDF files and extract Shari'ah rules.
- **Request:** `multipart/form-data` with SS PDFs and metadata.
- **Response Example:**
  ```json
  {
     "status": "success",
     "message": "SRMA processing complete.",
     "output_file_path": "output_srma_api/shariah_rules_mined_combined.json",
     "num_files_processed": 2
  }
  ```

---

## 7. 🧪 Example API Calls (cURL)

Replace file paths with actual files on your machine.

### 🟠 Initializing the System

```bash
curl -X POST \
  -F "fas_files=@dummy_fas.pdf" \
  -F "ss_files=@dummy_ss.pdf" \
  -F "shariah_rules_explicit_file=@shariah_rules_explicit.json" \
  -F "persist_db_path_base=./curl_test_db" \
  http://localhost:5001/initialize
```

### 🟡 Analyzing a Text Chunk

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "target_text_chunk": "An entity shall recognise revenue when (or as) the entity satisfies a performance obligation by transferring a promised good or service (i.e., an asset) to a customer.",
  "fas_context_chunks": [
     "Revenue is income arising in the course of an entity’s ordinary activities.",
     "A performance obligation is a promise in a contract with a customer to transfer to the customer either a good or service (or a bundle of goods or services) that is distinct."
  ],
  "ss_context_chunks": [
     "Contracts of exchange (Mu’awadat) must be free from major uncertainty (Gharar) and gambling (Maysir).",
     "The subject matter of a sale must be in existence, owned by the seller, and capable of delivery."
  ],
  "fas_name_for_display": "FAS on Revenue Recognition (Conceptual)",
  "identified_ambiguity": "The term asset transfer might need clarification on constructive transfer or control for Shariah compliance in specific Islamic finance contracts."
}' \
http://localhost:5001/analyze_chunk
```

### 🟣 Mining Shari'ah Rules

```bash
curl -X POST \
  -F "ss_files_for_srma=@dummy_ss_for_srma.pdf" \
  -F "ss_files_for_srma_0_fullname=Dummy Shariah Standard for SRMA" \
  -F "ss_files_for_srma_0_shortcode=DSSRM1" \
  -F "output_directory=./srma_curl_output" \
  http://localhost:5001/mine_shariah_rules
```

### 🟢 Checking System Status

```bash
curl http://localhost:5001/status
```

---

## 8. 🗂️ Project Structure

```
asave_api_project/
├── api_server.py               # Flask API application
├── agents/                     # Directory for AI agent classes
│   ├── __init__.py
│   ├── base_agent.py
│   ├── extraction_agent.py
│   ├── suggestion_agent.py
│   ├── shariah_rule_miner_agent.py
│   └── validation_agent.py
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

## 9. ⚠️ Important Considerations

- 🔑 **API Key:** The `GOOGLE_API_KEY` environment variable must be set for the AI agents to function.
- 🛡️ **Security:** For production, implement authentication, input validation, and run behind a WSGI server and reverse proxy with HTTPS.
- 📝 **Error Handling:** The API includes basic error handling. For production, add robust error management and logging.
- 🗃️ **State Management:** The `asave_context` dictionary holds the global state. For multi-process environments, consider alternatives.
- 💻 **Resource Intensive:** Initializing the system (especially processing PDFs and creating embeddings) can be resource-intensive.
- 👨‍⚖️ **Human Oversight:** All AI-generated outputs must be reviewed by qualified human experts.
- 📁 **File Paths:** Ensure file paths provided in API calls are correct relative to where the API server is running.
- 📝 **Logging:** The API uses Python's logging module. Check the console output for detailed logs.

---

> This `README.md` provides a solid starting point for users and developers of your ASAVE API. Replace placeholders like `<your-repo-url>` and ensure file paths in cURL examples are correct for your setup. Consider adding a "Troubleshooting" section as you develop further. 🚀

