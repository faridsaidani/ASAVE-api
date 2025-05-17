# 🚀 ASAVE (AAOIFI Standard Augmentation & Validation Engine)

ASAVE is a comprehensive platform designed to leverage AI for reviewing, suggesting, validating, and enhancing AAOIFI Financial Accounting Standards (FAS) and Shari'ah Standards (SS). It employs a multi-agent architecture powered by Google Gemini Large Language Models to process standards documents, provide detailed analysis, and ensure Shari'ah compliance.

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [API Endpoints](#api-endpoints)
5. [Prerequisites](#prerequisites)
6. [Setup & Installation](#setup--installation)
7. [Running the API Server](#running-the-api-server)
8. [API Usage Examples](#api-usage-examples)
9. [Document Processing and Session Management](#document-processing-and-session-management)
10. [Agent System Workflow](#agent-system-workflow)
11. [Project Structure](#project-structure)
12. [Error Handling](#error-handling)
13. [Important Considerations](#important-considerations)

---

## 🌟 Project Overview

The ASAVE system is built with Flask and orchestrates multiple specialized AI agents powered by Google Gemini LLMs via Langchain. It processes AAOIFI standards (in PDF format), a curated Shari'ah knowledge base, and user-provided text to:

- 💡 **Generate Suggestions**: Create clear, precise enhancements for AAOIFI Financial Accounting Standards.
- ✅ **Validate Compliance**: Verify suggestions against explicit Shari'ah rules and principles from Shari'ah Standards.
- 🔄 **Consistency Checking**: Ensure inter-standard consistency across the AAOIFI framework.
- 🏷️ **Extract Shari'ah Rules**: Mine and structure Shari'ah rules from AAOIFI Shari'ah Standards.
- 📊 **Document Analysis**: Extract, reformat, and analyze financial standard text with contextual understanding.

The system is designed with an emphasis on transparency, providing detailed JSON responses that include the reasoning process of each agent in the pipeline. This "thinking process" view allows users to understand how conclusions were reached and enhances trust in the AI-assisted recommendations.

---

## 🔍 Features

### Core Capabilities

- 📄 **Advanced Document Processing**
  - PDF text extraction with structural preservation
  - Intelligent text chunking with semantic boundaries
  - Marker-based pipeline for handling complex document layouts
  - AI-powered Markdown reformatting

- 🧠 **Vector Store Integration**
  - ChromaDB-based embeddings for semantic search
  - Efficient Retrieval Augmented Generation (RAG)
  - Persistent vector databases with session management

- 🤖 **AI Agent Ecosystem**
  - **AISGA (AI-Powered Suggestion Generation Agent)**: Proposes modifications or new clauses
  - **ConcisenessAgent**: Refines verbose content into clear, precise language
  - **ValidationAgent (SCVA)**: Performs Shari'ah compliance validation
  - **ValidationAgent (ISCCA)**: Checks inter-standard consistency
  - **ShariahRuleMinerAgent (SRMA)**: Extracts structured Shari'ah rules
  - **TextReformatterAgent**: Enhances document readability and structure
  - **ContextualUpdateAgent**: Updates standards based on new regulatory context

### System Features

- 📝 **Session Management**
  - Multi-session support with unique session IDs
  - Document versioning within sessions
  - Stateful processing across API calls

- 📊 **Format Handling**
  - JSONL to JSON conversion with robust error handling
  - PDF text extraction with formatting preservation
  - Structured data export (JSON, JSONL, CSV)

- 🌐 **Interactive API**
  - RESTful endpoints with comprehensive documentation
  - Streaming responses for long-running operations
  - Detailed JSON outputs with agent reasoning traces

---

## 🏗️ System Architecture

ASAVE follows a modular architecture centered around a Flask API server that coordinates multiple specialized AI agents:

```
                           ┌─────────────────┐
                           │   Flask API     │
                           │    Server       │
                           └───────┬─────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
         ┌───────▼─────┐   ┌───────▼─────┐   ┌───────▼─────┐
         │  Document   │   │    Agent    │   │   Vector    │
         │  Processor  │   │ Orchestrator│   │   Stores    │
         └───────┬─────┘   └───────┬─────┘   └───────┬─────┘
                 │                 │                 │
┌────────────────┼─────────────────┼─────────────────┼────────────────┐
│                │                 │                 │                │
│     ┌──────────▼──────────┐      │      ┌──────────▼──────────┐     │
│     │  PDF Text Extraction│◄─────┘─────►│   Suggestion Agent  │     │
│     └───────────┬─────────┘              └───────────┬────────┘     │
│                 │                                    │              │
│     ┌───────────▼─────────┐              ┌───────────▼────────┐     │
│     │ Document Chunking   │              │  Validation Agent  │     │
│     └───────────┬─────────┘              └───────────┬────────┘     │
│                 │                                    │              │
│     ┌───────────▼─────────┐              ┌───────────▼────────┐     │
│     │   Text Reformatting │              │ Shariah Rule Miner │     │
│     └─────────────────────┘              └────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components:

1. **API Server**: Flask-based RESTful API handling client requests
2. **Document Processor**: Handles PDF extraction, text chunking, and reformatting
3. **Agent Orchestrator**: Coordinates AI agent workflow and manages interaction
4. **Vector Stores**: ChromaDB-based embedding storage for semantic retrieval
5. **Agent System**: Specialized AI agents for different tasks (suggestion, validation, etc.)
6. **Session Manager**: Maintains state and manages document versions

---

## 🔌 API Endpoints

### Core Endpoints

- **GET /status**  
  _Returns current operational status of the ASAVE system_
  
  **Response:**
  ```json
  {
    "status": "operational",
    "initialized": true,
    "version": "1.0.0",
    "available_agents": ["AISGA", "SCVA", "ISCCA", "SRMA"],
    "loaded_resources": {
      "fas_documents": ["FAS 1", "FAS 28"],
      "ss_documents": ["SS 1", "SS 17"],
      "explicit_rules": 42
    }
  }
  ```

- **POST /initialize**  
  _Initializes the ASAVE system with necessary documents and configuration_
  
  **Request Body:**
  - `fas_files`: PDF file(s) containing Financial Accounting Standards
  - `ss_files`: PDF file(s) containing Shari'ah Standards
  - `shariah_rules_explicit_file`: JSON/JSONL file with explicit Shari'ah rules
  - `persist_db_path_base` (optional): Path for persisting vector databases
  - `session_id` (optional): Custom session identifier
  
  **Response:**
  ```json
  {
    "status": "success",
    "message": "ASAVE system initialized successfully",
    "session_id": "session_2023-05-01_12-30-45_789",
    "processing_details": {
      "fas_documents_processed": 2,
      "ss_documents_processed": 1,
      "explicit_rules_loaded": 42,
      "vector_db_location": "./db_store_api/session_2023-05-01_12-30-45_789"
    }
  }
  ```

- **POST /analyze_chunk**  
  _Analyzes a specific text chunk with context for suggestions and validation_
  
  **Request Body:**
  ```json
  {
    "target_text_chunk": "An entity shall recognise revenue when (or as) the entity satisfies a performance obligation by transferring a promised good or service (i.e., an asset) to a customer.",
    "fas_context_chunks": [
      "Revenue is income arising in the course of an entity's ordinary activities.",
      "A performance obligation is a promise in a contract with a customer to transfer to the customer either a good or service (or a bundle of goods or services) that is distinct."
    ],
    "ss_context_chunks": [
      "Contracts of exchange (Mu'awadat) must be free from major uncertainty (Gharar) and gambling (Maysir).",
      "The subject matter of a sale must be in existence, owned by the seller, and capable of delivery."
    ],
    "fas_name_for_display": "FAS on Revenue Recognition",
    "identified_ambiguity": "The term asset transfer might need clarification on constructive transfer or control for Shariah compliance in specific Islamic finance contracts."
  }
  ```
  
  **Response:**
  ```json
  {
    "status": "success",
    "analysis_results": {
      "original_text": "An entity shall recognise revenue when (or as) the entity satisfies a performance obligation by transferring a promised good or service (i.e., an asset) to a customer.",
      "suggestion_agent": {
        "proposed_text": "An entity shall recognise revenue when (or as) the entity satisfies a performance obligation by transferring control of a promised good or service (i.e., an asset) to a customer, ensuring the transfer is free from major uncertainty (Gharar) and the asset exists and is capable of delivery.",
        "confidence_score": 0.85,
        "enhancement_reasoning": "The original text was enhanced by adding 'control of' to clarify the nature of transfer, and explicit reference to 'free from major uncertainty (Gharar)' and asset existence to align with Shari'ah principles from the context."
      },
      "validation_agent": {
        "overall_status": "Needs Expert Review",
        "summary_explanation": "The suggestion appears broadly compliant but expert review is recommended for the construction of 'control transfer' in Islamic contexts.",
        "explicit_rule_batch_assessment": {
          "status_from_llm": "No Violations Found",
          "identified_issues": []
        },
        "semantic_validation_against_ss": {
          "status": "Aligned",
          "notes": "The proposed text aligns with Shari'ah principles regarding Gharar avoidance and certainty in transactions."
        }
      },
      "inter_standard_consistency": {
        "status": "Consistent",
        "explanation": "The terminology and principles in the proposed text align with other FAS documents covering revenue recognition and asset transfer concepts."
      }
    },
    "processing_time_ms": 2345
  }
  ```

- **POST /mine_shariah_rules**  
  _Extracts and structures Shari'ah rules from SS documents_
  
  **Request Body:**
  - `ss_files_for_srma`: PDF file(s) containing Shari'ah Standards for mining
  - `ss_files_for_srma_0_fullname`: Full name of the first Shari'ah Standard
  - `ss_files_for_srma_0_shortcode`: Short code identifier for the first Shari'ah Standard
  - `output_directory` (optional): Custom directory for output files
  
  **Response:**
  ```json
  {
    "status": "success",
    "message": "Shariah rules mining completed successfully",
    "results": {
      "documents_processed": 1,
      "rules_extracted": 25,
      "output_location": "./output_srma_api/DWSRM1_rules.json",
      "sample_rules": [
        {
          "rule_id": "DWSRM1-001",
          "description": "It is not permissible to trade in debt securities.",
          "standard_ref": "Dummy Shariah Standard for SRMA - Section 2.3.1",
          "principle_keywords": ["debt", "trading", "prohibition"]
        }
      ]
    }
  }
  ```

### Document Processing Endpoints

- **POST /extract_text_from_pdf**  
  _Extracts text from a PDF file with optional AI reformatting_
  
  **Request Body:**
  - `pdf_file`: The PDF file to extract text from
  - `ai_reformat` (optional): Boolean flag for AI reformatting (default: false)
  
  **Response:**
  ```json
  {
    "status": "success",
    "extracted_text": "# Financial Accounting Standard 1\n\n## General Presentation and Disclosure\n\n### Scope\n\nThis standard applies to all financial statements prepared in accordance with Islamic accounting principles...",
    "pages_processed": 10,
    "reformatted": true
  }
  ```

- **POST /extract_text_from_pdf_file_marker**  
  _Extracts text and images from a PDF using the marker pipeline_
  
  **Request Body:**
  - `pdf_file`: The PDF file to process
  
  **Response:**
  ```json
  {
    "status": "success",
    "extracted_content": {
      "text": "...",
      "sections": [
        {
          "title": "Introduction",
          "level": 1,
          "content": "..."
        }
      ],
      "images": [
        {
          "page": 1,
          "dimensions": "300x200",
          "location": "./temp_api_uploads/images/img_p1_001.png"
        }
      ]
    }
  }
  ```

### Session Management Endpoints

- **POST /create_session**  
  _Creates a new session for document processing_
  
  **Request Body:**
  - `session_name` (optional): Custom name for the session
  
  **Response:**
  ```json
  {
    "status": "success",
    "session_id": "session_2023-05-01_12-30-45_789",
    "expiry": "2023-05-08T12:30:45Z"
  }
  ```

- **GET /sessions**  
  _Lists all active sessions_
  
  **Response:**
  ```json
  {
    "status": "success",
    "active_sessions": [
      {
        "session_id": "session_2023-05-01_12-30-45_789",
        "created_at": "2023-05-01T12:30:45Z",
        "document_count": 3,
        "versions": 2
      }
    ]
  }
  ```

- **POST /convert_jsonl_to_json**  
  _Converts JSONL file to JSON format_
  
  **Request Body:**
  - `jsonl_file`: The JSONL file to convert
  
  **Response:**
  ```json
  {
    "status": "success",
    "converted_data": [...],
    "source_lines": 42,
    "target_objects": 42
  }
  ```

---

## 🛠️ Prerequisites

- 🐍 **Python 3.9+** (add to system PATH)
- 📦 **pip** (Python package manager)
- 🔑 **Google Gemini API Key** (from Google AI Studio or Google Cloud)
- 📄 **AAOIFI FAS and SS documents** in PDF format
- 🗂️ **shariah_rules_explicit.json** file (optional, for pre-defined rules)
- 🧪 **Postman** (recommended for API testing)
- 💻 **curl** (for command-line API interaction)

---

## 📥 Setup & Installation

### Windows

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd asave_project
   ```

2. **Set Google API Key:**
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

3. **Create a Virtual Environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

   **Example `requirements.txt`:**
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
   ```

5. **Prepare Input Files:**
   - Place AAOIFI FAS and SS PDFs in an accessible folder
   - Place or create `shariah_rules_explicit.json` in the project root

### Linux/macOS

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd asave_project
   ```

2. **Set Google API Key:**
   ```bash
   export GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
   ```
   - To set permanently, add to `~/.bashrc` or `~/.zshrc`

3. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare Input Files:**
   - Place AAOIFI FAS and SS PDFs in an accessible folder
   - Place or create `shariah_rules_explicit.json` in the project root

---

## 🚀 Running the API Server

Once setup is complete, run the Flask API server from your activated virtual environment:

### Windows
```cmd
python api_server.py
```

### Linux/macOS
```bash
python3 api_server.py
```

- 🌐 By default, the server runs at [http://localhost:5001](http://localhost:5001)
- 🗂️ The server automatically creates several directories:
  - `temp_api_uploads/`: Temporary storage for uploaded files
  - `db_store_api/`: ChromaDB vector stores
  - `output_srma_api/`: Output files from SRMA
  - `asave_sessions_db/`: Session data storage
  - `pdf_library/`: PDF document storage

### Server Logs
Monitor the console output for detailed logging information during operation.

---

## 🧪 API Usage Examples

### Initializing the System

#### Using Postman
1. Import the ASAVE collection
2. Set `{{baseUrl}}` to `http://localhost:5001`
3. Navigate to "Initialize System" request
4. Use "Select Files" for file parameters
5. Click "Send"

#### Using curl (Windows CMD)
```cmd
curl -X POST ^
  -F "fas_files=@C:/path/to/fas_document.pdf" ^
  -F "ss_files=@C:/path/to/shariah_standard.pdf" ^
  -F "shariah_rules_explicit_file=@C:/path/to/shariah_rules_explicit.json" ^
  -F "persist_db_path_base=./my_vector_db" ^
  http://localhost:5001/initialize
```

#### Using curl (Linux/macOS)
```bash
curl -X POST \
  -F "fas_files=@/path/to/fas_document.pdf" \
  -F "ss_files=@/path/to/shariah_standard.pdf" \
  -F "shariah_rules_explicit_file=@/path/to/shariah_rules_explicit.json" \
  -F "persist_db_path_base=./my_vector_db" \
  http://localhost:5001/initialize
```

### Analyzing a Text Chunk

#### Using Postman
1. Navigate to "Analyze Text Chunk" request
2. Edit the JSON body with your target text and context
3. Click "Send"

#### Using curl
1. Save the following JSON to `payload_analyze.json`:
```json
{
  "target_text_chunk": "An entity shall recognise revenue when (or as) the entity satisfies a performance obligation by transferring a promised good or service (i.e., an asset) to a customer.",
  "fas_context_chunks": [
    "Revenue is income arising in the course of an entity's ordinary activities.",
    "A performance obligation is a promise in a contract with a customer to transfer to the customer either a good or service (or a bundle of goods or services) that is distinct."
  ],
  "ss_context_chunks": [
    "Contracts of exchange (Mu'awadat) must be free from major uncertainty (Gharar) and gambling (Maysir).",
    "The subject matter of a sale must be in existence, owned by the seller, and capable of delivery."
  ],
  "fas_name_for_display": "FAS on Revenue Recognition",
  "identified_ambiguity": "The term asset transfer might need clarification on constructive transfer or control for Shariah compliance in specific Islamic finance contracts."
}
```

2. Run:
```cmd
curl -X POST -H "Content-Type: application/json" -d "@payload_analyze.json" http://localhost:5001/analyze_chunk
```

### Mining Shari'ah Rules

#### Using Postman
1. Navigate to "Mine Shari'ah Rules (SRMA)" request
2. Select SS document file(s)
3. Fill in standard name and shortcode
4. Click "Send"

#### Using curl (Windows CMD)
```cmd
curl -X POST ^
  -F "ss_files_for_srma=@C:/path/to/shariah_standard.pdf" ^
  -F "ss_files_for_srma_0_fullname=AAOIFI Shari'ah Standard on Sale of Debt" ^
  -F "ss_files_for_srma_0_shortcode=SS59" ^
  -F "output_directory=./my_rules_output" ^
  http://localhost:5001/mine_shariah_rules
```

### Extract Text from PDF with AI Reformatting

#### Using curl (Windows CMD)
```cmd
curl -X POST ^
  -F "pdf_file=@C:/path/to/document.pdf" ^
  -F "ai_reformat=true" ^
  http://localhost:5001/extract_text_from_pdf
```

### Converting JSONL to JSON

#### Using curl (Windows CMD)
```cmd
curl -X POST ^
  -F "jsonl_file=@C:/path/to/rules.jsonl" ^
  http://localhost:5001/convert_jsonl_to_json
```

---

## 📑 Document Processing and Session Management

### Document Processing Pipeline

The ASAVE system employs a sophisticated document processing pipeline:

1. **PDF Extraction**: Using PyMuPDF (fitz) to extract raw text
2. **Text Chunking**: Intelligent segmentation with:
   - Semantic boundary preservation
   - Configurable overlap
   - Special handling for headers/sections

3. **Vector Store Creation**:
   - Text embedding generation
   - ChromaDB persistent storage
   - Metadata tagging (source, page numbers, etc.)

4. **AI Enhancement**:
   - Reformatting to improve structure
   - Markdown conversion
   - Table and list identification

### Session Management

ASAVE uses a comprehensive session management system:

1. **Session Creation**:
   - Unique session IDs (timestamp-based or custom)
   - Session metadata storage
   - Expiration policies

2. **Document Versioning**:
   - Version tracking for modified documents
   - Comparison between versions
   - Rollback capabilities

3. **State Persistence**:
   - SQLite database for session data
   - File-based vector store persistence
   - Automatic cleanup of expired sessions

Example session database schema:
```sql
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  created_at TIMESTAMP,
  expires_at TIMESTAMP,
  metadata TEXT
);

CREATE TABLE documents (
  doc_id TEXT PRIMARY KEY,
  session_id TEXT,
  filename TEXT,
  doc_type TEXT,
  current_version INTEGER,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE document_versions (
  version_id INTEGER PRIMARY KEY,
  doc_id TEXT,
  version_num INTEGER,
  created_at TIMESTAMP,
  vector_store_path TEXT,
  metadata TEXT,
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
```

---

## 🤖 Agent System Workflow

The ASAVE multi-agent system coordinates several specialized AI agents:

### AISGA (AI-Powered Suggestion Generation Agent)

**Purpose**: Generate suggestions to enhance FAS text clarity and compliance  
**Input**: Target text chunk, FAS context, identified ambiguity  
**Output**: Proposed text, confidence score, enhancement reasoning

**Example interaction**:
```
Input: "An entity shall recognise revenue when the entity satisfies a performance obligation."

Output: {
  "proposed_text": "An entity shall recognise revenue when the entity satisfies a performance obligation by transferring control of a promised asset, ensuring the transaction is free from Gharar.",
  "confidence_score": 0.87,
  "enhancement_reasoning": "Added specificity about transfer of control and Shari'ah compliance requirement"
}
```

### SCVA (Shari'ah Compliance Validation Agent)

**Purpose**: Validate suggestions against Shari'ah rules  
**Input**: Proposed text, explicit Shari'ah rules, SS context  
**Output**: Compliance status, identified issues, semantic validation

**Processing workflow**:
1. Batch validation against explicit rules
2. Semantic validation against SS vector store
3. Overall compliance determination
4. Detailed explanation and issue identification

### SRMA (Shari'ah Rule Miner Agent)

**Purpose**: Extract structured Shari'ah rules from SS documents  
**Input**: SS document text  
**Output**: Structured rules with IDs, descriptions, and metadata

**Rule extraction process**:
1. Document chunking with overlap
2. Rule identification in each chunk
3. Rule structuring and deduplication
4. JSON/JSONL output formatting

### ConcisenessAgent

**Purpose**: Transform verbose text into clear, concise language  
**Input**: Lengthy text with potential redundancies  
**Output**: Concise text preserving core meaning and compliance

### Other Specialized Agents

- **TextReformatterAgent**: Enhances document readability
- **ContextualUpdateAgent**: Updates standards based on new context
- **Agents Collaboration**: Coordinated workflow with agent chaining

---

## 📁 Project Structure

```
asave_project/
├── api_server.py              # Main API server entry point
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── agents/                    # AI agent implementations
│   ├── __init__.py           
│   ├── base_agent.py          # Abstract base agent class
│   ├── suggestion_agent.py    # AISGA implementation
│   ├── validation_agent.py    # SCVA & ISCCA implementation
│   ├── shariah_rule_miner_agent.py  # SRMA implementation
│   ├── conciseness_agent.py   # Text conciseness agent
│   ├── contextual_update_agent.py   # Handles regulatory updates
│   └── text_reformatter_agent.py    # Document reformatting
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── document_processor.py  # PDF handling and text processing
│   ├── vector_store_manager.py  # ChromaDB interface
│   └── session_manager.py     # Session handling
├── marker/                    # Marker pipeline components
│   ├── converters/            # Format converters
│   ├── models/                # ML models config
│   ├── output/                # Output formatters
│   └── config/                # Configuration parsers
├── temp_api_uploads/          # Temporary file storage
├── db_store_api/              # Vector database storage
├── output_srma_api/           # Rule mining output
├── asave_sessions_db/         # Session database
└── pdf_library/               # Document storage
```

---

## 🐞 Error Handling

ASAVE implements comprehensive error handling mechanisms:

### JSONL Parsing Errors

The system includes robust handling for JSONL files with malformed lines:

```python
def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
    rules = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//'):  # Skip comments and empty lines
                    try:
                        rule = json.loads(line)
                        rules.append(rule)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line in {file_path}: {line[:50]}... - {e}")
        return rules
    except Exception as e:
        logger.error(f"Error loading JSONL file {file_path}: {e}")
        return []
```

### Input Sanitization

Text inputs are sanitized to prevent prompt injection or template errors:

```python
def _sanitize_text(self, text: str) -> str:
    """Escape curly braces and other problematic characters for prompt templates."""
    if not text:
        return ""
    return text.replace("{", "{{").replace("}", "}}")
```

### API Error Responses

All API endpoints return standardized error formats:

```json
{
  "status": "error",
  "error_type": "ValidationError",
  "message": "Missing required parameter: fas_files",
  "details": {
    "parameter": "fas_files",
    "required": true
  }
}
```

### LLM Error Handling

The system gracefully handles LLM errors:

```python
try:
    llm_response = self.invoke_chain(chain, input_data)
    # Process response
except Exception as e:
    logger.error(f"LLM call failed: {str(e)}", exc_info=True)
    return {
        "status": "Error",
        "explanation": f"Error during LLM processing: {str(e)}",
        "fallback_response": default_response
    }
```
