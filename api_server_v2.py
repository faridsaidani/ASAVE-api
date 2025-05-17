# api_server.py
import inspect
import os
import json
import logging
import shutil
import time # For simulating work or adding deliberate delays
import fitz
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import sqlite3
import datetime
import os
import shutil
import json
import logging
import hashlib



# ASAVE Core Components
from utils.document_processor import DocumentProcessor
from agents.suggestion_agent import SuggestionAgent # AISGA
from agents.conciseness_agent import ConcisenessAgent
from agents.validation_agent import ValidationAgent
from agents.shariah_rule_miner_agent import ShariahRuleMinerAgent
from agents.text_reformatter_agent import TextReformatterAgent # <-- NEW IMPORT
from agents.contextual_update_agent import ContextualUpdateAgent 
from flask_cors import CORS

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import os
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_api_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'json'}
CORS(app, resources={r"/*": {"origins": "*"}})
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CONFIGURED_PDF_LIBRARY_PATH = os.path.join(app.root_path, 'pdf_library') 
os.makedirs(CONFIGURED_PDF_LIBRARY_PATH, exist_ok=True) # Ensure it exists
DOCUMENT_VERSIONS_DIR_NAME = "doc_versions" # Subdirectory within a session for document versions


# Directory to save session-specific vector databases
SESSIONS_DB_PATH = os.path.join(app.root_path, 'asave_sessions_db')
os.makedirs(SESSIONS_DB_PATH, exist_ok=True)
DATABASE_NAME = "asave_document_versions.db"
DATABASE_PATH = os.path.join(app.root_path, DATABASE_NAME) # Store DB at app root or SESSIONS_DB_PATH


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

MAX_PARALLEL_AGENT_WORKERS = 3 # Adjust as needed
executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENT_WORKERS)

asave_context = {
    "doc_processor": None,
    "fas_vector_store": None,
    "ss_vector_store": None,
    "all_fas_vector_store": None,
    "aisga_variants": {},
    "specialized_agents": {},
    "scva_iscca": None,
    "srma": None,
    "shariah_rules_explicit_path": "shariah_rules_explicit.json",
    "mined_shariah_rules_path": "output_srma_api/shariah_rules_mined_combined.json",
    "initialized": False,
    "text_reformatter_agent": None,
    "text_reformatter_marker" : None,
    "current_session_id": None, # Name or ID of the currently loaded session
    "initialized_with_session": False, # Tracks if current init is from a saved session
    "cua_agent": None,
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Helper for SSE ---
def stream_event(data_dict):
    """Formats a dictionary into an SSE message string."""
    return f"data: {json.dumps(data_dict)}\n\n"

# --- Helper for parallel agent execution ---
def process_suggestion_task(agent_instance, method_name, variant_name_override=None, **kwargs):
    actual_variant_name = variant_name_override or getattr(agent_instance, 'agent_type', type(agent_instance).__name__)
    try:
        if hasattr(agent_instance, method_name):
            method_to_call = getattr(agent_instance, method_name)
            if "variant_name" in method_to_call.__code__.co_varnames:
                kwargs["variant_name"] = actual_variant_name
            return method_to_call(**kwargs)
        else:
            logger.error(f"Method {method_name} not found in agent {type(agent_instance).__name__}")
            return {"error": f"Method {method_name} not found.", "agent_type": type(agent_instance).__name__}
    except Exception as e:
        logger.error(f"Exception in agent {type(agent_instance).__name__} ({actual_variant_name}) calling {method_name}: {e}", exc_info=True)
        return {"error": str(e), "agent_type": type(agent_instance).__name__, "variant_name": actual_variant_name}
def get_document_version_path(session_id: str, document_id: str) -> str:
    """Constructs the base path for storing versions of a specific document within a session."""
    # Sanitize document_id to be a safe directory name
    safe_document_id = secure_filename(document_id.replace('.pdf', '').replace('.md', ''))
    if not safe_document_id: # Handle empty or weird document IDs
        safe_document_id = "untitled_document"
    path = os.path.join(SESSIONS_DB_PATH, session_id, DOCUMENT_VERSIONS_DIR_NAME, safe_document_id)
    os.makedirs(path, exist_ok=True) # Ensure directory exists
    return path

def get_active_document_path(session_id: str, document_id: str) -> str:
    """Path to where the current active version of the Markdown is stored."""
    # This could be a specific file, or you might decide the latest version IS the active one.
    # For simplicity, let's assume the main Markdown content the user edits is stored directly
    # in the session, and versions are copies. Or, the "active" is just a pointer to latest version.
    # Let's make it a distinct file for the current editable content.
    safe_document_id = secure_filename(document_id.replace('.pdf', '').replace('.md', ''))
    if not safe_document_id: safe_document_id = "untitled_document"
    active_doc_dir = os.path.join(SESSIONS_DB_PATH, session_id, "active_docs")
    os.makedirs(active_doc_dir, exist_ok=True)
    return os.path.join(active_doc_dir, f"{safe_document_id}.md")

def get_current_timestamp_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Microsecond precision for uniqueness

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Documents Table: Tracks unique documents within sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_pkid INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            original_document_id TEXT NOT NULL, -- User-facing ID, e.g., PDF filename
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            current_version_fk INTEGER, -- Points to the version_pkid of the active version
            notes TEXT,
            UNIQUE(session_id, original_document_id),
            FOREIGN KEY (current_version_fk) REFERENCES document_versions(version_pkid) ON DELETE SET NULL
        )
    ''')
    # Document Versions Table: Tracks each version of a document's content
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_versions (
            version_pkid INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_fk INTEGER NOT NULL,
            version_timestamp_id TEXT NOT NULL UNIQUE, -- e.g., YYYYMMDD_HHMMSS_ffffff
            content_filepath TEXT NOT NULL,          -- Relative path to the .md file in session's version dir
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            change_summary TEXT,
            content_hash TEXT, -- MD5 or SHA256 hash of the content
            parent_version_fk INTEGER, -- Points to the previous version_pkid (for lineage)
            FOREIGN KEY (doc_fk) REFERENCES documents(doc_pkid) ON DELETE CASCADE,
            FOREIGN KEY (parent_version_fk) REFERENCES document_versions(version_pkid) ON DELETE SET NULL 
        )
    ''')
    # Index for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_fk_created_at ON document_versions (doc_fk, created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_session_original_id ON documents (session_id, original_document_id)")

    conn.commit()
    conn.close()
    logger.info(f"Database '{DATABASE_NAME}' initialized/checked at {DATABASE_PATH}")

def get_or_create_document_record(conn: sqlite3.Connection, session_id: str, original_document_id: str) -> int:
    """Gets the PKID of a document record, creating it if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_pkid FROM documents WHERE session_id = ? AND original_document_id = ?",
        (session_id, original_document_id)
    )
    row = cursor.fetchone()
    if row:
        return row["doc_pkid"]
    else:
        cursor.execute(
            "INSERT INTO documents (session_id, original_document_id) VALUES (?, ?)",
            (session_id, original_document_id)
        )
        conn.commit()
        logger.info(f"Created new document record for session '{session_id}', doc '{original_document_id}', PKID: {cursor.lastrowid}")
        return cursor.lastrowid # type: ignore

# --- Non-Streaming (Standard JSON) API Endpoints ---

@app.route('/initialize', methods=['POST'])
def initialize_asave():
    global asave_context
    logger.info("Received /initialize request.")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Initialize failed: GOOGLE_API_KEY not set.")
        return jsonify({"status": "error", "message": "GOOGLE_API_KEY environment variable not set."}), 500
    
    try:
        data = request.form 
        uploaded_fas_files = request.files.getlist('fas_files_upload')
        uploaded_ss_files = request.files.getlist('ss_files_upload')
        uploaded_rules_file = request.files.get('shariah_rules_explicit_file_upload')
        session_name_to_save = data.get('save_as_session_name')
        session_id_to_load = data.get('load_session_id')
        library_fas_filenames_json = data.get('library_fas_filenames')
        library_ss_filenames_json = data.get('library_ss_filenames')
        
        current_session_persist_path = None; effective_session_id = None

        if session_id_to_load:
            effective_session_id = secure_filename(session_id_to_load)
            current_session_persist_path = os.path.join(SESSIONS_DB_PATH, effective_session_id)
            if not os.path.isdir(current_session_persist_path): return jsonify({"status": "error", "message": f"Session '{effective_session_id}' not found."}), 404
            asave_context["current_session_id"] = effective_session_id; asave_context["initialized_with_session"] = True
        elif session_name_to_save:
            effective_session_id = secure_filename(session_name_to_save)
            current_session_persist_path = os.path.join(SESSIONS_DB_PATH, effective_session_id)
            if os.path.exists(current_session_persist_path) and not data.get('overwrite_session', 'false').lower() == 'true':
                return jsonify({"status": "error", "message": f"Session name '{effective_session_id}' already exists."}), 400
            os.makedirs(current_session_persist_path, exist_ok=True)
            asave_context["current_session_id"] = effective_session_id; asave_context["initialized_with_session"] = False
        else: 
            effective_session_id = f"default_temp_session_{get_current_timestamp_str()}"; current_session_persist_path = os.path.join(SESSIONS_DB_PATH, effective_session_id)
            os.makedirs(current_session_persist_path, exist_ok=True)
            asave_context["current_session_id"] = effective_session_id; asave_context["initialized_with_session"] = False
        
        asave_context["doc_processor"] = DocumentProcessor()

        session_rules_path = os.path.join(current_session_persist_path, "shariah_rules_explicit.json")
        if uploaded_rules_file and allowed_file(uploaded_rules_file.filename):
            uploaded_rules_file.save(session_rules_path); asave_context["shariah_rules_explicit_path"] = session_rules_path
        elif os.path.exists(asave_context["shariah_rules_explicit_path"]): # Default global path
            if not os.path.exists(session_rules_path): shutil.copy2(asave_context["shariah_rules_explicit_path"], session_rules_path)
            asave_context["shariah_rules_explicit_path"] = session_rules_path
        else: 
            with open(session_rules_path, "w") as f: json.dump([{"rule_id": f"DUMMY_{effective_session_id}_001", "description":"Dummy rule", "validation_query_template": "Is valid?"}], f)
            asave_context["shariah_rules_explicit_path"] = session_rules_path

        fas_filepaths_to_process = []; ss_filepaths_to_process = []
        upload_temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"session_uploads_{time.time_ns()}")
        os.makedirs(upload_temp_dir, exist_ok=True)
        for file in uploaded_fas_files:
            if file and allowed_file(file.filename): filepath = os.path.join(upload_temp_dir, secure_filename(file.filename)); file.save(filepath); fas_filepaths_to_process.append(filepath)
        for file in uploaded_ss_files:
            if file and allowed_file(file.filename): filepath = os.path.join(upload_temp_dir, secure_filename(file.filename)); file.save(filepath); ss_filepaths_to_process.append(filepath)
        if library_fas_filenames_json:
            for name in json.loads(library_fas_filenames_json): path = os.path.join(CONFIGURED_PDF_LIBRARY_PATH, secure_filename(name)); fas_filepaths_to_process.append(path)
        if library_ss_filenames_json:
            for name in json.loads(library_ss_filenames_json): path = os.path.join(CONFIGURED_PDF_LIBRARY_PATH, secure_filename(name)); ss_filepaths_to_process.append(path)
        
        fas_db_path = os.path.join(current_session_persist_path, "fas_db"); ss_db_path = os.path.join(current_session_persist_path, "ss_db")
        doc_processor = asave_context["doc_processor"]

        if session_id_to_load:
            asave_context["fas_vector_store"] = doc_processor.load_vector_store(fas_db_path)
            asave_context["ss_vector_store"] = doc_processor.load_vector_store(ss_db_path)
            if not asave_context["fas_vector_store"] and not fas_filepaths_to_process : logger.warning(f"Failed to load FAS store for {effective_session_id} and no new files.")
            if not asave_context["ss_vector_store"] and not ss_filepaths_to_process : logger.warning(f"Failed to load SS store for {effective_session_id} and no new files.")
            asave_context["all_fas_vector_store"] = asave_context["fas_vector_store"] 

        if not asave_context.get("fas_vector_store") and fas_filepaths_to_process:
            all_fas_chunks = []; 
            for fp in fas_filepaths_to_process: docs = doc_processor.load_pdf(fp); chunks = doc_processor.chunk_text(docs); all_fas_chunks.extend(chunks)
            if all_fas_chunks: asave_context["fas_vector_store"] = doc_processor.create_vector_store(all_fas_chunks, persist_directory=fas_db_path); asave_context["all_fas_vector_store"] = asave_context["fas_vector_store"]
        if not asave_context.get("ss_vector_store") and ss_filepaths_to_process:
            all_ss_chunks = [];
            for fp in ss_filepaths_to_process: docs = doc_processor.load_pdf(fp); chunks = doc_processor.chunk_text(docs); all_ss_chunks.extend(chunks)
            if all_ss_chunks: asave_context["ss_vector_store"] = doc_processor.create_vector_store(all_ss_chunks, persist_directory=ss_db_path)
        
        if os.path.exists(upload_temp_dir): shutil.rmtree(upload_temp_dir)

        logger.info("Initializing AI Agents for current session...")
        asave_context["scva_iscca"] = ValidationAgent()
        asave_context["srma"] = ShariahRuleMinerAgent()
        asave_context["aisga_variants"]["pro_conservative_detailed"] = SuggestionAgent(model_name="gemini-1.5-pro-latest", temperature=0.2); asave_context["aisga_variants"]["pro_conservative_detailed"].agent_type = "AISGA_ProConsDetailed"
        asave_context["aisga_variants"]["flash_alternative_options"] = SuggestionAgent(model_name="gemini-1.5-flash-latest", temperature=0.7); asave_context["aisga_variants"]["flash_alternative_options"].agent_type = "AISGA_FlashCreative"
        asave_context["specialized_agents"]["conciseness_agent"] = ConcisenessAgent()
        try: asave_context["cua_agent"] = ContextualUpdateAgent(); logger.info("Initialized ContextualUpdateAgent (CUA)")
        except Exception as e_cua: logger.error(f"CUA Init Error: {e_cua}", exc_info=True)
        
        # Marker Init
        raw_config_marker = {"gemini_api_key": os.getenv("GOOGLE_API_KEY"), "output_format": "markdown", "paginate_output": True, "use_llm": True} # Use LLM for better structuring
        config_parser_marker = ConfigParser(raw_config_marker)
        asave_context["text_reformatter_marker"] = PdfConverter(config=config_parser_marker.generate_config_dict(), artifact_dict=create_model_dict(), processor_list=config_parser_marker.get_processors(), renderer=config_parser_marker.get_renderer(), llm_service=config_parser_marker.get_llm_service())
        
        asave_context["text_reformatter_agent"] = TextReformatterAgent() # For PyMuPDF+AI method

        asave_context["initialized"] = True
        logger.info(f"ASAVE system initialized. Session: '{effective_session_id}'.")
        return jsonify({ "status": "success", "message": f"ASAVE initialized for session '{effective_session_id}'.", "session_id": effective_session_id, "fas_vector_store_status": "Ready" if asave_context.get("fas_vector_store") else "Not Available", "ss_vector_store_status": "Ready" if asave_context.get("ss_vector_store") else "Not Available" }), 200

    except ValueError as ve: # Catches API key issues from agent constructors or DocProcessor
        logger.error(f"Initialization ValueError: {ve}", exc_info=True); asave_context["initialized"] = False
        return jsonify({"status": "error", "message": f"Core Component Initialization Error: {str(ve)}"}), 500
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True); asave_context["initialized"] = False
        return jsonify({"status": "error", "message": f"General initialization error: {str(e)}"}), 500

@app.route('/list_sessions', methods=['GET'])
def list_sessions_api():
    # ... (same as before) ...
    logger.info("Received /list_sessions request.")
    try:
        sessions = []
        if os.path.exists(SESSIONS_DB_PATH):
            for item_name in os.listdir(SESSIONS_DB_PATH):
                item_path = os.path.join(SESSIONS_DB_PATH, item_name)
                if os.path.isdir(item_path):
                    fas_db_exists = os.path.exists(os.path.join(item_path, "fas_db")); ss_db_exists = os.path.exists(os.path.join(item_path, "ss_db"))
                    if fas_db_exists or ss_db_exists: sessions.append({ "session_id": item_name, "path": item_path, "has_fas_db": fas_db_exists, "has_ss_db": ss_db_exists, "last_modified": time.ctime(os.path.getmtime(item_path)) if os.path.exists(item_path) else "N/A" })
        return jsonify({"status": "success", "sessions": sessions}), 200
    except Exception as e: logger.error(f"Error listing sessions: {e}", exc_info=True); return jsonify({"status": "error", "message": f"Failed to list sessions: {str(e)}"}), 500


@app.route('/status', methods=['GET'])
def get_status_api():
    global asave_context
    return jsonify({
        "service_status": "running", "asave_initialized": asave_context["initialized"],
        "current_session_id": asave_context.get("current_session_id"),
        "config": { "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY")), "upload_folder": app.config['UPLOAD_FOLDER'], "explicit_rules_path_in_session": asave_context["shariah_rules_explicit_path"], "mined_rules_path_default_location": asave_context["mined_shariah_rules_path"] },
        "components_loaded": { "doc_processor": bool(asave_context["doc_processor"]), "fas_vector_store": bool(asave_context["fas_vector_store"]), "ss_vector_store": bool(asave_context["ss_vector_store"]), "num_aisga_variants": len(asave_context["aisga_variants"]), "num_specialized_agents": len(asave_context["specialized_agents"]), "scva_iscca": bool(asave_context["scva_iscca"]), "srma": bool(asave_context["srma"]), "cua_agent": bool(asave_context.get("cua_agent")), "text_reformatter_agent": bool(asave_context.get("text_reformatter_agent")), "text_reformatter_marker": bool(asave_context.get("text_reformatter_marker")) }
    })

# process_agent_task helper (critical for orchestrating agent calls)
def process_agent_task(agent_instance, method_name: str, task_description: str, **kwargs) -> dict:
    agent_class_name = type(agent_instance).__name__
    # Use 'variant_name' from kwargs if provided (for AISGA variants), else use agent_type or class name
    specific_agent_name = kwargs.get("variant_name", getattr(agent_instance, 'agent_type', agent_class_name))
    
    full_task_id = f"Task: {task_description} (Agent: {specific_agent_name}, Method: {method_name})"
    logger.info(f"EXECUTING {full_task_id} with args: {list(k for k in kwargs.keys() if k != 'ss_context_strings' and k != 'fas_context_strings')}") # Log args, but not long context strings

    response_payload = {
        "status": "error", "task": task_description, "agent_type": agent_class_name,
        "agent_name_variant": specific_agent_name, "result": None, "error": "Unknown error."
    }
    try:
        if not agent_instance: response_payload["error"] = "Agent instance is None."; logger.error(f"FAILED {full_task_id}. Reason: Agent instance is None."); return response_payload
        if not hasattr(agent_instance, method_name): response_payload["error"] = f"Method '{method_name}' not found."; logger.error(f"FAILED {full_task_id}. Reason: Method not found."); return response_payload
        
        method_to_call = getattr(agent_instance, method_name)
        sig = inspect.signature(method_to_call); call_kwargs = {}
        for param_name, param_obj in sig.parameters.items():
            if param_name == 'self': continue
            if param_name in kwargs: call_kwargs[param_name] = kwargs[param_name]
        
        agent_result = method_to_call(**call_kwargs)

        if isinstance(agent_result, dict) and "error" in agent_result:
            response_payload["error"] = agent_result["error"]
            if "llm_response_received" in agent_result: response_payload["llm_response_on_agent_error"] = agent_result["llm_response_received"]
            if "prompt_details_actual" in agent_result: response_payload["prompt_details_actual_on_agent_error"] = agent_result["prompt_details_actual"]
            logger.warning(f"Task PARTIALLY FAILED (agent reported error): {full_task_id}. Agent Error: {agent_result['error']}")
        else:
            response_payload["status"] = "success"; response_payload["result"] = agent_result; response_payload["error"] = None
            logger.info(f"COMPLETED {full_task_id} successfully.")
    except TypeError as te: response_payload["error"] = f"TypeError: {te}. Check arguments."; logger.error(f"FAILED {full_task_id}. Reason: {response_payload['error']}", exc_info=True)
    except Exception as e: response_payload["error"] = f"General exception: {e}"; logger.error(f"FAILED {full_task_id}. Reason: {response_payload['error']}", exc_info=True)
    return response_payload


@app.route('/get_assistance_stream', methods=['POST'])
def get_assistance_stream_api():
    # ... (same as previous version, ensuring process_agent_task is used for AISGA calls,
    # and the resulting suggestion_payload which contains confidence_score is passed into
    # the validated_suggestion_package's suggestion_details field) ...
    global asave_context, executor
    logger.info("Received /get_assistance_stream request.")
    if not asave_context["initialized"] or not asave_context["scva_iscca"]: 
        def e_init(): yield stream_event({"event_type": "fatal_error", "message":"System not ready."}); 
        return Response(stream_with_context(e_init()),mimetype='text/event-stream', status=503)
    try:
        data = request.get_json(); selected_text = data.get("selected_text_from_fas"); fas_doc_id = data.get("fas_document_id")
        if not selected_text or not fas_doc_id: 
            def e_payload(): yield stream_event({"event_type": "fatal_error", "message":"Missing payload fields."}); 
            return Response(stream_with_context(e_payload()),mimetype='text/event-stream', status=400)
    except Exception as e_req: 
        def e_json(): yield stream_event({"event_type": "fatal_error", "message":f"Request error: {e_req}."}); 
        return Response(stream_with_context(e_json()),mimetype='text/event-stream', status=400)

    def generate_analysis_stream_content():
        try:
            yield stream_event({"event_type": "system_log", "step_code": "STREAM_START", "message": "FAS analysis stream initiated..."})
            # Retrieve SS Context & FAS Context (using RAG)
            ss_context_strings = []; fas_context_strings = [] # Populate these via RAG
            if asave_context.get("ss_vector_store"): 
                retriever_ss = asave_context["ss_vector_store"].as_retriever(search_kwargs={"k": 5})
                ss_context_strings = [doc.page_content for doc in retriever_ss.get_relevant_documents(selected_text)]
                yield stream_event({"event_type": "progress", "step_code": "SS_CTX_DONE", "message": f"Retrieved {len(ss_context_strings)} SS snippets."})
            if asave_context.get("fas_vector_store"):
                retriever_fas = asave_context["fas_vector_store"].as_retriever(search_kwargs={"k": 3}) # Assuming fas_vector_store is session-specific
                fas_context_strings = [doc.page_content for doc in retriever_fas.get_relevant_documents(selected_text)]
                yield stream_event({"event_type": "progress", "step_code": "FAS_CTX_DONE", "message": f"Retrieved {len(fas_context_strings)} local FAS snippets."})

            suggestion_futures = {}
            for variant_name, aisga_instance in asave_context["aisga_variants"].items():
                if aisga_instance:
                    yield stream_event({"event_type": "progress", "step_code": f"AISGA_START_{variant_name.upper()}", "agent_name": variant_name, "message": f"AISGA '{variant_name}' drafting..."})
                    future = executor.submit( process_agent_task, aisga_instance, "generate_clarification", task_description=f"AISGA_{variant_name}_FAS_clarify", original_text=selected_text, identified_ambiguity="User selection from FAS.", fas_context_strings=fas_context_strings, ss_context_strings=ss_context_strings, variant_name=variant_name )
                    suggestion_futures[future] = {"type": "AISGA", "name": variant_name}
            # Conciseness agent (if configured to produce suggestions for this flow)
            conciseness_agent = asave_context["specialized_agents"].get("conciseness_agent")
            if conciseness_agent:
                 yield stream_event({"event_type": "progress", "step_code": "CONCISENESS_START", "agent_name": "ConcisenessAgent", "message": "Conciseness Agent working..."})
                 future_concise = executor.submit(process_agent_task, conciseness_agent, "make_concise", task_description="ConciseAgent_FAS_edit", text_to_make_concise=selected_text, shariah_context_strings=ss_context_strings)
                 suggestion_futures[future_concise] = {"type": "Specialized", "name": "ConcisenessAgent"}


            raw_suggestions_for_validation = []
            for future in as_completed(suggestion_futures):
                agent_info = suggestion_futures[future]; task_result_wrapper = future.result()
                if task_result_wrapper["status"] == "success" and task_result_wrapper.get("result"):
                    raw_suggestions_for_validation.append({"source_agent_info": agent_info, "suggestion_payload": task_result_wrapper["result"]}) # payload has confidence
                    yield stream_event({"event_type": "agent_suggestion_generated", "agent_name": agent_info['name'], "message": f"Suggestion from '{agent_info['name']}'.", "payload": {"summary": str(task_result_wrapper["result"].get("proposed_text",""))[:70]+"...", "confidence": task_result_wrapper["result"].get("confidence_score")}})
                else: yield stream_event({"event_type": "warning", "agent_name": agent_info['name'], "message": f"Agent '{agent_info['name']}' failed: {task_result_wrapper.get('error', 'Empty result')}"})

            validation_futures_map = {}
            for item in raw_suggestions_for_validation:
                sugg_payload = item["suggestion_payload"]; agent_info = item["source_agent_info"]
                yield stream_event({"event_type": "progress", "agent_name": agent_info['name'], "message": f"Validating suggestion from '{agent_info['name']}'..."})
                scva_future = executor.submit(asave_context["scva_iscca"].validate_shariah_compliance_batched, proposed_suggestion_object=sugg_payload, shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"], ss_vector_store=asave_context["ss_vector_store"], mined_shariah_rules_path=asave_context["mined_shariah_rules_path"])
                iscca_future = executor.submit(asave_context["scva_iscca"].validate_inter_standard_consistency, proposed_suggestion_object=sugg_payload, fas_name=fas_doc_id, all_fas_vector_store=asave_context["all_fas_vector_store"])
                validation_futures_map[(scva_future, iscca_future)] = item
            
            total_validated_count = 0
            for (scva_future, iscca_future), original_item_info in validation_futures_map.items():
                agent_name = original_item_info["source_agent_info"]["name"]
                try:
                    scva_report = scva_future.result(); iscca_report = iscca_future.result() # Assuming these are the direct results from agent methods
                    validated_suggestion_package = { "source_agent_type": original_item_info["source_agent_info"]["type"], "source_agent_name": agent_name, "suggestion_details": original_item_info["suggestion_payload"], "scva_report": scva_report, "iscca_report": iscca_report, "validation_summary_score": f"SCVA: {scva_report.get('overall_status', 'N/A')}, ISCCA: {iscca_report.get('status', 'N/A')}" }
                    total_validated_count += 1
                    yield stream_event({"event_type": "validated_suggestion_package", "agent_name": agent_name, "message": f"Validation complete for '{agent_name}'.", "payload": validated_suggestion_package })
                except Exception as e_val: yield stream_event({"event_type": "error", "agent_name": agent_name, "message": f"Validation collection error for '{agent_name}': {e_val}"})
            yield stream_event({"event_type": "system_log", "step_code": "STREAM_END", "message": "FAS analysis stream finished.", "payload": {"total_validated_suggestions": total_validated_count}})
        except Exception as e_stream: logger.error(f"FAS Stream Error: {e_stream}", exc_info=True); yield stream_event({"event_type": "fatal_error", "message": f"Stream failed: {e_stream}"})
    return Response(stream_with_context(generate_analysis_stream_content()), mimetype='text/event-stream')


@app.route('/extract_text_from_pdf', methods=['POST'])
def extract_text_from_pdf_api(): # PyMuPDF + Optional AI Reformatting (TextReformatterAgent)
    # ... (same logic as before, uses asave_context["text_reformatter_agent"]) ...
    logger.info("Received /extract_text_from_pdf request.")
    if 'pdf_file' not in request.files: return jsonify({"status": "error", "message": "No 'pdf_file'."}), 400
    file = request.files['pdf_file']
    if file.filename == '': return jsonify({"status": "error", "message": "No selected file."}), 400
    reformatter = asave_context.get("text_reformatter_agent")
    apply_ai_reformat = request.args.get('reformat_ai', 'true').lower() == 'true'
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename); pdf_bytes = file.read(); doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_data = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num); raw_text = page.get_text("text", sort=True).strip()
                if reformatter and apply_ai_reformat and raw_text:
                    reformat_result = reformatter.reformat_to_markdown(raw_text, page_number=page_num + 1)
                    pages_data.append({"page_number": page_num + 1, "content": reformat_result["markdown_content"] if reformat_result["status"] == "success" else raw_text, "reformatted": reformat_result["status"] == "success"})
                else: pages_data.append({"page_number": page_num + 1, "content": raw_text, "reformatted": False})
            doc.close()
            return jsonify({"status": "success", "document_info": {"filename":filename, "page_count":len(pages_data)}, "pages": pages_data}), 200
        except Exception as e: logger.error(f"PDF processing error: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Invalid file type."}), 400


@app.route('/extract_text_from_pdf_file_marker', methods=['POST'])
def extract_text_from_pdf_file_marker_api(): # Marker-based extraction
    # ... (same logic as before, uses asave_context["text_reformatter_marker"]) ...
    logger.info("Received /extract_text_from_pdf_file_marker request.")
    if 'pdf_file' not in request.files: return jsonify({"status": "error", "message": "No 'pdf_file'."}), 400
    file = request.files['pdf_file']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"status": "error", "message": "No file or invalid type."}), 400
    converter = asave_context.get("text_reformatter_marker")
    if not converter: return jsonify({"status": "error", "message": "Marker PDF converter not initialized."}), 503
    try:
        filename = secure_filename(file.filename); temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(temp_path)
        rendered_output = converter(temp_path); text_content, _, _ = text_from_rendered(rendered_output) # Ignoring images for now
        try: os.remove(temp_path)
        except Exception as e_clean: logger.warning(f"Failed to remove temp marker file {temp_path}: {e_clean}")
        return jsonify({"status": "success", "filename": filename, "extracted_text": text_content, "document_info": {"filename": filename, "source": "marker"}}), 200
    except Exception as e: logger.error(f"Marker PDF extraction error: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/validate_contract_terms_stream', methods=['POST'])
def validate_contract_terms_stream_api():
    # ... (same as previous version, ensuring process_agent_task is used for AISGA calls,
    # and the resulting suggestion_payload which contains confidence_score is passed into
    # the clause_ai_suggestion_generated event's payload's suggestion_details field) ...
    global asave_context, executor
    logger.info("Received /validate_contract_terms_stream request.")
    if not asave_context.get("initialized") or not asave_context.get("scva_iscca"): 
        def e_init(): yield stream_event({"event_type": "fatal_error", "message":"System not ready for contract validation."}); 
        return Response(stream_with_context(e_init()),mimetype='text/event-stream', status=503)
    try:
        data = request.get_json(); contract_type = data.get("contract_type", "General"); client_clauses_raw = data.get("client_clauses"); overall_ctx = data.get("overall_contract_context", "")
        if not client_clauses_raw or not isinstance(client_clauses_raw, list): 
            def e_payload(): yield stream_event({"event_type": "fatal_error", "message":"Invalid 'client_clauses' payload."}); 
            return Response(stream_with_context(e_payload()),mimetype='text/event-stream', status=400)
        client_clauses = [c for c in client_clauses_raw if isinstance(c, dict) and c.get("text","").strip()]
        if not client_clauses : 
            def e_noclause(): yield stream_event({"event_type": "fatal_error", "message":"No valid clauses provided."}); 
            return Response(stream_with_context(e_noclause()),mimetype='text/event-stream', status=400)
    except Exception as e_req: 
        def e_json(): yield stream_event({"event_type": "fatal_error", "message":f"Request parsing error: {e_req}."}); 
        return Response(stream_with_context(e_json()),mimetype='text/event-stream', status=400)

    def generate_contract_validation_stream():
        try:
            yield stream_event({"event_type": "system_log", "step_code": "CONTRACT_STREAM_START", "message": f"Contract validation initiated for '{contract_type}'..."})
            ss_context_for_all_clauses = [] # Populate via RAG if ss_vector_store is available
            if asave_context.get("ss_vector_store"):
                combined_clause_text = " ".join([c['text'][:100] for c in client_clauses])
                query_for_ss = f"{contract_type} {overall_ctx} {combined_clause_text}"
                retriever_ss = asave_context["ss_vector_store"].as_retriever(search_kwargs={"k": 7})
                ss_context_for_all_clauses = [doc.page_content for doc in retriever_ss.get_relevant_documents(query_for_ss)]
                yield stream_event({"event_type": "progress", "step_code": "SS_CTX_CONTRACT_DONE", "message": f"Retrieved {len(ss_context_for_all_clauses)} general SS snippets."})

            for i, clause_item in enumerate(client_clauses):
                clause_id = clause_item.get("clause_id", f"user_clause_{i+1}_{time.time_ns()}")
                clause_text = clause_item["text"]
                yield stream_event({"event_type": "clause_processing_start", "payload": {"clause_id": clause_id, "original_text": clause_text, "message": f"Analyzing clause ({clause_id})..."}})
                
                # SCVA for original client clause
                scva_future = executor.submit(process_agent_task, asave_context["scva_iscca"], "validate_shariah_compliance_batched", task_description=f"SCVA_client_{clause_id}", proposed_suggestion_object={"proposed_text": clause_text, "shariah_notes": f"Client clause for {contract_type}. Context: {overall_ctx}"}, shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"], ss_vector_store=asave_context["ss_vector_store"], mined_shariah_rules_path=asave_context["mined_shariah_rules_path"], contract_type=contract_type)
                scva_result_wrapper = scva_future.result()
                client_clause_scva_report = scva_result_wrapper.get("result") if scva_result_wrapper["status"] == "success" else {"error": scva_result_wrapper.get("error", "SCVA failed")}
                yield stream_event({"event_type": "clause_validation_result", "payload": {"clause_id": clause_id, "original_text": clause_text, "scva_report": client_clause_scva_report }})

                # AISGA Suggestions
                needs_ai_suggestion = True # Or based on client_clause_scva_report.get("overall_status")
                if needs_ai_suggestion:
                    aisga_futures = {}
                    ambiguity_desc_contract = f"Client clause for {contract_type}. SCVA: {client_clause_scva_report.get('overall_status', 'N/A')}. Please review/enhance."
                    for variant, aisga in asave_context["aisga_variants"].items():
                        future = executor.submit(process_agent_task, aisga, "generate_clarification", task_description=f"AISGA_{variant}_contract_{clause_id}", original_text=clause_text, identified_ambiguity=ambiguity_desc_contract, fas_context_strings=[], ss_context_strings=ss_context_for_all_clauses, variant_name=variant)
                        aisga_futures[future] = {"type": "AISGA", "name": variant, "clause_id": clause_id}
                    
                    for future_sugg in as_completed(aisga_futures):
                        sugg_agent_info = aisga_futures[future_sugg]; sugg_wrapper = future_sugg.result()
                        if sugg_wrapper["status"] == "success" and sugg_wrapper.get("result"):
                            ai_sugg_payload = sugg_wrapper["result"] # This includes confidence_score
                            # SCVA for AI's suggestion
                            scva_ai_future = executor.submit(process_agent_task, asave_context["scva_iscca"], "validate_shariah_compliance_batched", task_description=f"SCVA_AISugg_{sugg_agent_info['name']}_{clause_id}", proposed_suggestion_object=ai_sugg_payload, shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"], ss_vector_store=asave_context["ss_vector_store"], mined_shariah_rules_path=asave_context["mined_shariah_rules_path"], contract_type=contract_type)
                            scva_ai_report = scva_ai_future.result().get("result") if scva_ai_future.result()["status"] == "success" else {"error": scva_ai_future.result().get("error", "SCVA for AI sugg failed")}
                            
                            packaged_suggestion = { "clause_id": clause_id, "source_agent_type": sugg_agent_info["type"], "source_agent_name": sugg_agent_info["name"], "suggestion_details": ai_sugg_payload, "scva_report": scva_ai_report, "validation_summary_score": f"AI Sugg SCVA: {scva_ai_report.get('overall_status', 'N/A')}" } # scva_report is now for AI's suggestion
                            yield stream_event({"event_type": "clause_ai_suggestion_generated", "payload": packaged_suggestion})
                        else: yield stream_event({"event_type": "error", "agent_name": sugg_agent_info['name'], "message": f"AISGA '{sugg_agent_info['name']}' for clause {clause_id} failed: {sugg_wrapper.get('error', 'Empty')}"})
                yield stream_event({"event_type": "clause_processing_end", "payload": {"clause_id": clause_id, "message": f"Finished clause {clause_id}."}})
            yield stream_event({"event_type": "system_log", "step_code": "CONTRACT_STREAM_END", "message": "Contract terms validation stream finished."})
        except Exception as e_stream: logger.error(f"Contract Stream Error: {e_stream}", exc_info=True); yield stream_event({"event_type": "fatal_error", "message": f"Contract stream failed: {e_stream}"})
    return Response(stream_with_context(generate_contract_validation_stream()), mimetype='text/event-stream')

@app.route('/review_full_contract_stream', methods=['POST'])
def review_full_contract_stream_api():
    # ... (same logic as before, ValidationAgent.review_entire_contract will be called via process_agent_task) ...
    global asave_context, executor
    logger.info("Received /review_full_contract_stream request.")
    if not asave_context.get("initialized") or not asave_context.get("scva_iscca"): 
        def e_init(): yield stream_event({"event_type": "fatal_error", "message":"System not ready for full contract review."}); 
        return Response(stream_with_context(e_init()),mimetype='text/event-stream', status=503)
    try:
        data = request.get_json(); full_contract_text = data.get("full_contract_text"); contract_type = data.get("contract_type", "General Contract")
        if not full_contract_text or not isinstance(full_contract_text, str) or not full_contract_text.strip():
            def e_payload(): yield stream_event({"event_type": "fatal_error", "message":"Missing or empty 'full_contract_text'."}); 
            return Response(stream_with_context(e_payload()),mimetype='text/event-stream', status=400)
    except Exception as e_req: 
        def e_json(): yield stream_event({"event_type": "fatal_error", "message":f"Request parsing error: {e_req}."}); 
        return Response(stream_with_context(e_json()),mimetype='text/event-stream', status=400)

    validation_agent = asave_context["scva_iscca"]
    def generate_full_review_stream():
        try:
            yield stream_event({"event_type": "system_log", "step_code": "FULL_CONTRACT_REVIEW_START", "message": f"Full contract review initiated for type '{contract_type}'..."})
            yield stream_event({"event_type": "progress", "step_code": "AI_FULL_REVIEW_IN_PROGRESS", "message": "🤖 AI is performing holistic review. This may take time..."})
            
            review_future = executor.submit( process_agent_task, validation_agent, "review_entire_contract", task_description=f"FullReview_{contract_type}", contract_text=full_contract_text, contract_type=contract_type, shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"], mined_shariah_rules_path=asave_context["mined_shariah_rules_path"], ss_vector_store=asave_context.get("ss_vector_store") )
            review_result_wrapper = review_future.result()

            if review_result_wrapper["status"] == "success" and review_result_wrapper.get("result"):
                yield stream_event({"event_type": "full_contract_review_completed", "payload": review_result_wrapper["result"]})
            else:
                yield stream_event({"event_type": "error", "step_code": "AI_FULL_REVIEW_FAILED", "message": f"AI review failed: {review_result_wrapper.get('error', 'Unknown error')}", "payload": {"error_details": review_result_wrapper.get('error')}})
            yield stream_event({"event_type": "system_log", "step_code": "FULL_CONTRACT_STREAM_END", "message": "Full contract review stream finished."})
        except Exception as e_stream: logger.error(f"Full Contract Review Stream Error: {e_stream}", exc_info=True); yield stream_event({"event_type": "fatal_error", "message": f"Full contract stream failed: {e_stream}"})
    return Response(stream_with_context(generate_full_review_stream()), mimetype='text/event-stream')


@app.route('/list_library_pdfs', methods=['GET'])
def list_library_pdfs_api():
    logger.info("Received /list_library_pdfs request.")
    try:
        pdf_files = []
        # Scan for subdirectories (e.g., 'FAS', 'SS') or just list all PDFs
        for item in os.listdir(CONFIGURED_PDF_LIBRARY_PATH):
            item_path = os.path.join(CONFIGURED_PDF_LIBRARY_PATH, item)
            if os.path.isfile(item_path) and item.lower().endswith('.pdf'):
                pdf_files.append({"name": item, "type": "file"})
            elif os.path.isdir(item_path): # Basic subdirectory listing
                subdir_files = []
                for sub_item in os.listdir(item_path):
                    if sub_item.lower().endswith('.pdf') and os.path.isfile(os.path.join(item_path, sub_item)):
                         subdir_files.append(sub_item)
                if subdir_files:
                    pdf_files.append({"name": item, "type": "directory", "files": subdir_files})
        
        # More sophisticated: categorize by subfolder (e.g., FAS, SS)
        # For now, just a flat list or simple dir structure
        
        return jsonify({"status": "success", "library_path": CONFIGURED_PDF_LIBRARY_PATH, "pdf_files": pdf_files}), 200
    except Exception as e:
        logger.error(f"Error listing library PDFs: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to list library PDFs: {str(e)}"}), 500

@app.route('/document/<session_id>/<path:document_id>/save_version', methods=['POST'])
def save_document_version_api_db(session_id: str, document_id: str): # document_id can now contain slashes if it's a path
    logger.info(f"Received /save_version (DB) for session '{session_id}', doc '{document_id}'")
    session_id = secure_filename(session_id) # Basic sanitization
    # document_id is kept as is, as it might represent a path/name from library

    if not asave_context.get("initialized") or asave_context.get("current_session_id") != session_id:
        return jsonify({"status": "error", "message": f"Session '{session_id}' not active or initialized."}), 400

    data = request.get_json()
    if not data or "markdown_content" not in data:
        return jsonify({"status": "error", "message": "Missing 'markdown_content'."}), 400

    markdown_content = data["markdown_content"]
    change_summary = data.get("change_summary", "Version saved via API")
    parent_version_id_timestamp = data.get("parent_version_timestamp_id") # Optional: client can specify parent

    conn = None
    try:
        conn = get_db_connection()
        doc_pkid = get_or_create_document_record(conn, session_id, document_id)

        # Determine parent_version_fk if parent_version_id_timestamp is provided
        parent_version_fk = None
        if parent_version_id_timestamp:
            cursor_parent = conn.cursor()
            cursor_parent.execute(
                "SELECT version_pkid FROM document_versions WHERE doc_fk = ? AND version_timestamp_id = ?",
                (doc_pkid, parent_version_id_timestamp)
            )
            parent_row = cursor_parent.fetchone()
            if parent_row:
                parent_version_fk = parent_row["version_pkid"]
            else:
                logger.warning(f"Parent version timestamp ID '{parent_version_id_timestamp}' not found for doc_pkid {doc_pkid}. Saving without explicit parent link.")


        # 1. Save Markdown content to filesystem
        version_storage_base_path = get_document_version_path(session_id, document_id) # e.g., .../sessions/sid/doc_versions/docid/
        version_timestamp_id_str = get_current_timestamp_str()
        version_filename_md = f"version_{version_timestamp_id_str}.md"
        version_filepath_md_absolute = os.path.join(version_storage_base_path, version_filename_md)
        
        with open(version_filepath_md_absolute, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Markdown content saved to: {version_filepath_md_absolute}")

        # Relative path for DB storage (relative to SESSIONS_DB_PATH/session_id)
        # This makes the SESSIONS_DB_PATH potentially movable.
        sanitized_doc_id_for_path = secure_filename(document_id.replace('.pdf', '').replace('.md', '')) or "untitled_document"
        content_filepath_relative = os.path.join(DOCUMENT_VERSIONS_DIR_NAME, sanitized_doc_id_for_path, version_filename_md)
        content_hash_val = hashlib.md5(markdown_content.encode('utf-8')).hexdigest()

        # 2. Save version metadata to SQLite
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO document_versions 
                (doc_fk, version_timestamp_id, content_filepath, change_summary, content_hash, parent_version_fk)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_pkid, version_timestamp_id_str, content_filepath_relative, change_summary, content_hash_val, parent_version_fk))
        new_version_pkid = cursor.lastrowid

        # 3. Update the document record to point to this new version as the current one
        cursor.execute("UPDATE documents SET current_version_fk = ? WHERE doc_pkid = ?", (new_version_pkid, doc_pkid))
        conn.commit()
        
        # 4. Update the "active" document file (working copy)
        active_doc_file_path = get_active_document_path(session_id, document_id)
        shutil.copy2(version_filepath_md_absolute, active_doc_file_path)
        logger.info(f"Active document updated: {active_doc_file_path}")


        logger.info(f"New version {version_timestamp_id_str} (PKID: {new_version_pkid}) saved for doc_pkid {doc_pkid}")
        return jsonify({
            "status": "success", "message": "Document version saved successfully.",
            "version_id": version_timestamp_id_str, # User-friendly ID
            "version_pkid": new_version_pkid,     # Internal DB ID
            "document_pkid": doc_pkid
        }), 201

    except sqlite3.Error as e_sql:
        logger.error(f"SQLite error saving version: {e_sql}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"status": "error", "message": f"Database error: {str(e_sql)}"}), 500
    except Exception as e:
        logger.error(f"Error saving version: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"status": "error", "message": f"Failed to save version: {str(e)}"}), 500
    finally:
        if conn: conn.close()


@app.route('/document/<session_id>/<path:document_id>/versions', methods=['GET'])
def get_document_versions_api_db(session_id: str, document_id: str):
    logger.info(f"Received /versions (DB) for session '{session_id}', doc '{document_id}'")
    session_id = secure_filename(session_id)
    # document_id is kept as is

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Get doc_pkid first
        cursor.execute(
            "SELECT doc_pkid FROM documents WHERE session_id = ? AND original_document_id = ?",
            (session_id, document_id)
        )
        doc_row = cursor.fetchone()
        if not doc_row:
            return jsonify({"status": "success", "document_id": document_id, "versions": [], "message": "Document not found in this session."}), 200 # Or 404 if preferred

        doc_pkid = doc_row["doc_pkid"]
        cursor.execute("""
            SELECT version_timestamp_id, created_at, change_summary, content_filepath
            FROM document_versions 
            WHERE doc_fk = ? 
            ORDER BY created_at DESC
        """, (doc_pkid,))
        
        versions_data = []
        for row in cursor.fetchall():
            versions_data.append({
                "version_id": row["version_timestamp_id"],
                "timestamp": row["created_at"],
                "summary": row["change_summary"],
                "content_filepath_relative": row["content_filepath"] # For debug or direct access if needed
            })
        return jsonify({"status": "success", "document_id": document_id, "doc_pkid": doc_pkid, "versions": versions_data}), 200
    except sqlite3.Error as e_sql:
        logger.error(f"SQLite error listing versions: {e_sql}", exc_info=True)
        return jsonify({"status": "error", "message": f"Database error: {str(e_sql)}"}), 500
    except Exception as e:
        logger.error(f"Error listing versions: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to list versions: {str(e)}"}), 500
    finally:
        if conn: conn.close()


@app.route('/document/<session_id>/<path:document_id>/versions/<version_timestamp_id>', methods=['GET'])
def get_document_version_content_api_db(session_id: str, document_id: str, version_timestamp_id: str):
    logger.info(f"Received /versions/{version_timestamp_id} (DB) for session '{session_id}', doc '{document_id}'")
    session_id = secure_filename(session_id)
    version_timestamp_id = secure_filename(version_timestamp_id) # It's a timestamp string

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT dv.content_filepath, dv.change_summary, dv.created_at, d.original_document_id
            FROM document_versions dv
            JOIN documents d ON dv.doc_fk = d.doc_pkid
            WHERE d.session_id = ? AND d.original_document_id = ? AND dv.version_timestamp_id = ?
        """, (session_id, document_id, version_timestamp_id))
        
        row = cursor.fetchone()
        if row:
            # Construct absolute path to the MD file using SESSIONS_DB_PATH and session_id
            # content_filepath from DB is relative to SESSIONS_DB_PATH/session_id
            md_file_path_absolute = os.path.join(SESSIONS_DB_PATH, session_id, row["content_filepath"])
            if os.path.exists(md_file_path_absolute):
                with open(md_file_path_absolute, "r", encoding="utf-8") as f:
                    content = f.read()
                return jsonify({
                    "status": "success", "version_id": version_timestamp_id, "document_id": document_id,
                    "markdown_content": content,
                    "metadata": {"summary": row["change_summary"], "timestamp": row["created_at"]}
                }), 200
            else:
                logger.error(f"Markdown file not found at path from DB: {md_file_path_absolute}")
                return jsonify({"status": "error", "message": "Version content file missing on server."}), 404
        else:
            return jsonify({"status": "error", "message": "Version not found."}), 404
    except sqlite3.Error as e_sql: # ...
        return jsonify({"status": "error", "message": f"Database error: {str(e_sql)}"}), 500
    except Exception as e: # ...
        return jsonify({"status": "error", "message": f"Failed to get version content: {str(e)}"}), 500
    finally:
        if conn: conn.close()


@app.route('/document/<session_id>/<path:document_id>/revert_to_version', methods=['POST'])
def revert_to_version_api_db(session_id: str, document_id: str):
    logger.info(f"Received /revert_to_version (DB) for session '{session_id}', doc '{document_id}'")
    session_id = secure_filename(session_id)

    if not asave_context.get("initialized") or asave_context.get("current_session_id") != session_id:
         return jsonify({"status": "error", "message": f"Session '{session_id}' not active."}), 400

    data = request.get_json()
    if not data or "version_id_to_revert_to" not in data: # Match frontend key
        return jsonify({"status": "error", "message": "Missing 'version_id_to_revert_to'."}), 400
    
    target_version_timestamp_id = secure_filename(data["version_id_to_revert_to"])
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get doc_pkid and target version info
        cursor.execute("""
            SELECT dv.version_pkid, dv.content_filepath, d.doc_pkid
            FROM document_versions dv
            JOIN documents d ON dv.doc_fk = d.doc_pkid
            WHERE d.session_id = ? AND d.original_document_id = ? AND dv.version_timestamp_id = ?
        """, (session_id, document_id, target_version_timestamp_id))
        target_version_row = cursor.fetchone()

        if not target_version_row:
            return jsonify({"status": "error", "message": f"Target version '{target_version_timestamp_id}' not found."}), 404
        
        target_version_pkid = target_version_row["version_pkid"]
        target_content_filepath_relative = target_version_row["content_filepath"]
        doc_pkid = target_version_row["doc_pkid"]

        # 1. Read current active content (if any) to save it as a new version before reverting
        active_doc_file_path = get_active_document_path(session_id, document_id)
        current_active_content_for_backup = ""
        if os.path.exists(active_doc_file_path):
            with open(active_doc_file_path, "r", encoding="utf-8") as f_current:
                current_active_content_for_backup = f_current.read()
        
        # Save this current content as a new version (acts as a backup)
        backup_version_timestamp_id = get_current_timestamp_str()
        backup_filename_md = f"version_{backup_version_timestamp_id}.md"
        version_storage_base_path = get_document_version_path(session_id, document_id) # Base for version files
        backup_filepath_md_absolute = os.path.join(version_storage_base_path, backup_filename_md)
        
        with open(backup_filepath_md_absolute, "w", encoding="utf-8") as f_backup:
            f_backup.write(current_active_content_for_backup)
        
        sanitized_doc_id_for_path = secure_filename(document_id.replace('.pdf', '').replace('.md', '')) or "untitled_document"
        backup_content_filepath_relative = os.path.join(DOCUMENT_VERSIONS_DIR_NAME, sanitized_doc_id_for_path, backup_filename_md)
        backup_content_hash = hashlib.md5(current_active_content_for_backup.encode('utf-8')).hexdigest()

        # Get the current_version_fk to use as parent for this backup version
        cursor.execute("SELECT current_version_fk FROM documents WHERE doc_pkid = ?", (doc_pkid,))
        current_version_fk_row = cursor.fetchone()
        parent_for_backup_fk = current_version_fk_row["current_version_fk"] if current_version_fk_row else None

        cursor.execute("""
            INSERT INTO document_versions (doc_fk, version_timestamp_id, content_filepath, change_summary, content_hash, parent_version_fk)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_pkid, backup_version_timestamp_id, backup_content_filepath_relative, 
              f"Auto-save before reverting to version {target_version_timestamp_id}", backup_content_hash, parent_for_backup_fk))
        # We don't set this backup as current_version_fk for the document

        # 2. Update document's current_version_fk to the target version's PKID
        cursor.execute("UPDATE documents SET current_version_fk = ? WHERE doc_pkid = ?", (target_version_pkid, doc_pkid))

        # 3. Copy the target version's content to the active document file
        target_md_file_path_absolute = os.path.join(SESSIONS_DB_PATH, session_id, target_content_filepath_relative)
        if os.path.exists(target_md_file_path_absolute):
            shutil.copy2(target_md_file_path_absolute, active_doc_file_path)
            with open(active_doc_file_path, "r", encoding="utf-8") as f_reverted:
                reverted_content_for_response = f_reverted.read()
            conn.commit()
            logger.info(f"Reverted doc '{document_id}' to version '{target_version_timestamp_id}'.")
            return jsonify({
                "status": "success", "message": "Reverted to version successfully.",
                "reverted_markdown_content": reverted_content_for_response,
                "reverted_to_version_id": target_version_timestamp_id
            }), 200
        else:
            conn.rollback() # Important: rollback if content file is missing
            logger.error(f"Content file for target version not found: {target_md_file_path_absolute}")
            return jsonify({"status": "error", "message": "Target version content file missing on server."}), 500

    except sqlite3.Error as e_sql: # ...
        if conn: conn.rollback()
        return jsonify({"status": "error", "message": f"Database error: {str(e_sql)}"}), 500
    except Exception as e: # ...
        if conn: conn.rollback()
        return jsonify({"status": "error", "message": f"Failed to revert version: {str(e)}"}), 500
    finally:
        if conn: conn.close()


@app.route('/document/<session_id>/<path:document_id>/active_content', methods=['GET'])

def get_active_document_content_api_db(session_id: str, document_id: str): # document_id can be path-like
    logger.info(f"Received /active_content (DB) for session '{session_id}', doc '{document_id}'")
    session_id = secure_filename(session_id)
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Find the document and its current version
        cursor.execute("""
            SELECT dv.content_filepath, dv.version_timestamp_id, dv.change_summary, dv.created_at
            FROM documents d
            JOIN document_versions dv ON d.current_version_fk = dv.version_pkid
            WHERE d.session_id = ? AND d.original_document_id = ?
        """, (session_id, document_id))
        row = cursor.fetchone()

        if row:
            md_file_path_absolute = os.path.join(SESSIONS_DB_PATH, session_id, row["content_filepath"])
            if os.path.exists(md_file_path_absolute):
                with open(md_file_path_absolute, "r", encoding="utf-8") as f:
                    content = f.read()
                return jsonify({
                    "status": "success", "document_id": document_id, "markdown_content": content,
                    "current_version_info": {
                        "version_id": row["version_timestamp_id"],
                        "summary": row["change_summary"],
                        "timestamp": row["created_at"]
                    }
                }), 200
            else: # DB points to a file that doesn't exist - data inconsistency!
                 logger.error(f"Active content file missing for doc '{document_id}', session '{session_id}': {md_file_path_absolute}")
                 return jsonify({"status": "error", "message": "Active content file missing on server (data inconsistency)."}), 500
        else:
            # No current_version_fk set, or document doesn't exist.
            # Check if an "active_docs/doc.md" file exists as a fallback (e.g., from before DB or if current_version_fk is null)
            active_doc_legacy_path = get_active_document_path(session_id, document_id)
            if os.path.exists(active_doc_legacy_path):
                with open(active_doc_legacy_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return jsonify({"status": "success", "document_id": document_id, "markdown_content": content, "current_version_info": {"version_id": "N/A (legacy active file)"}}), 200
            else:
                return jsonify({"status": "success", "document_id": document_id, "markdown_content": "", "message": "No active content or versions found for this document."}), 200 # Or 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching active content: {str(e)}"}), 500
    finally:
        if conn: conn.close()

@app.route('/contextual_update/analyze', methods=['POST'])
def analyze_contextual_update_api():
    global asave_context
    logger.info("Received /contextual_update/analyze request.")

    if not asave_context.get("initialized") or not asave_context.get("current_session_id"):
        return jsonify({"status": "error", "message": "ASAVE system or session not initialized."}), 400
    
    cua: ContextualUpdateAgent = asave_context.get("cua_agent")
    if not cua:
        return jsonify({"status": "error", "message": "ContextualUpdateAgent not available."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

        new_context_text = data.get("new_context_text")
        target_document_id = data.get("target_document_id") # e.g., "FAS-17.pdf" or the ID used in versioning

        if not new_context_text or not target_document_id:
            return jsonify({"status": "error", "message": "Missing 'new_context_text' or 'target_document_id'."}), 400

        # --- Fetch current content of the target_document_id for the active session ---
        # This uses your existing document versioning logic to get the active/latest content.
        active_session_id = asave_context["current_session_id"]
        
        # Using get_active_document_content_api_db logic (simplified for direct call)
        # You might want to refactor this into a reusable function if not already.
        fas_document_content = ""
        conn = None
        try:
            conn = get_db_connection() # from your api_server
            cursor = conn.cursor()
            # Try to get from current_version_fk first
            cursor.execute("""
                SELECT dv.content_filepath
                FROM documents d
                JOIN document_versions dv ON d.current_version_fk = dv.version_pkid
                WHERE d.session_id = ? AND d.original_document_id = ?
            """, (active_session_id, target_document_id))
            row = cursor.fetchone()

            if row and row["content_filepath"]:
                md_file_path_absolute = os.path.join(SESSIONS_DB_PATH, active_session_id, row["content_filepath"])
                if os.path.exists(md_file_path_absolute):
                    with open(md_file_path_absolute, "r", encoding="utf-8") as f:
                        fas_document_content = f.read()
                else:
                    logger.warning(f"CUA: Active version file missing for {target_document_id} in session {active_session_id} at {md_file_path_absolute}. Checking legacy active_docs.")
            
            if not fas_document_content: # Fallback to legacy active_docs path if current_version_fk method failed
                active_doc_legacy_path = get_active_document_path(active_session_id, target_document_id) # from your api_server
                if os.path.exists(active_doc_legacy_path):
                    with open(active_doc_legacy_path, "r", encoding="utf-8") as f_legacy:
                        fas_document_content = f_legacy.read()
                        logger.info(f"CUA: Loaded content for {target_document_id} from legacy active_docs path.")
            
            if not fas_document_content:
                logger.error(f"CUA: Could not retrieve content for FAS document '{target_document_id}' in session '{active_session_id}'.")
                return jsonify({"status": "error", "message": f"Could not retrieve content for target FAS document '{target_document_id}'."}), 404

        except sqlite3.Error as e_sql_cua:
            logger.error(f"CUA: SQLite error retrieving document content: {e_sql_cua}", exc_info=True)
            return jsonify({"status": "error", "message": "Database error retrieving document content."}), 500
        finally:
            if conn: conn.close()
        
        # --- Invoke the CUA agent ---
        # This is a synchronous call as CUA's primary output is an analysis report.
        # For very long processing, you could make it async, but then the frontend
        # would need to poll or use a different SSE stream for CUA's results.
        
        # Running CUA in executor to prevent blocking main Flask thread for too long if LLM call is slow
        future_cua = executor.submit(
            cua.analyze_impact,
            new_context_text=new_context_text,
            fas_document_content=fas_document_content,
            fas_document_id=target_document_id
        )
        analysis_result = future_cua.result() # Wait for CUA to finish

        if "error" in analysis_result:
            return jsonify({"status": "error", "message": analysis_result["error"], "details": analysis_result}), 500

        return jsonify({"status": "success", "analysis": analysis_result}), 200

    except Exception as e:
        logger.error(f"Error during contextual update analysis: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"General error: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting ASAVE API server...")
    # Initialize the database and other components
    print("Initializing database...")
    initialize_database() # Initialize DB schema when app starts
    print("Database initialized.")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. API will not function correctly.")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True) # threaded=True is important for ThreadPoolExecutor with Flask dev server