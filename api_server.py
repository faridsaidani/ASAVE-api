# api_server.py
import inspect
import os
import json
import logging
import time # For simulating work or adding deliberate delays
import fitz
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed

# ASAVE Core Components
from utils.document_processor import DocumentProcessor
from agents.suggestion_agent import SuggestionAgent # AISGA
from agents.conciseness_agent import ConcisenessAgent
from agents.validation_agent import ValidationAgent
from agents.shariah_rule_miner_agent import ShariahRuleMinerAgent
from agents.text_reformatter_agent import TextReformatterAgent # <-- NEW IMPORT

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import os
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_api_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'json'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# --- Non-Streaming (Standard JSON) API Endpoints ---

@app.route('/initialize', methods=['POST'])
def initialize_asave():
    global asave_context
    logger.info("Received /initialize request.")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Initialize failed: GOOGLE_API_KEY not set.")
        return jsonify({"status": "error", "message": "GOOGLE_API_KEY environment variable not set."}), 500
    try:
        # --- File Handling and Basic Setup ---
        data = request.form
        fas_files_uploaded = request.files.getlist('fas_files')
        ss_files_uploaded = request.files.getlist('ss_files')
        shariah_rules_explicit_file_uploaded = request.files.get('shariah_rules_explicit_file')

        fas_filepaths = []
        for file in fas_files_uploaded:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                fas_filepaths.append(filepath)
        ss_filepaths = []
        for file in ss_files_uploaded:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                ss_filepaths.append(filepath)
        
        current_shariah_rules_explicit_path = asave_context["shariah_rules_explicit_path"]
        if shariah_rules_explicit_file_uploaded and allowed_file(shariah_rules_explicit_file_uploaded.filename):
            filename = secure_filename(shariah_rules_explicit_file_uploaded.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shariah_rules_explicit_file_uploaded.save(filepath)
            current_shariah_rules_explicit_path = filepath
            asave_context["shariah_rules_explicit_path"] = filepath
        elif not os.path.exists(current_shariah_rules_explicit_path):
            logger.warning(f"Explicit Shari'ah rules file not provided and default not found: {current_shariah_rules_explicit_path}. Creating dummy file.")
            dummy_rules = [{"rule_id": "DUMMY-API-INIT-001", "description":"Dummy rule", "validation_query_template": "Is {suggestion_text} compliant?"}]
            os.makedirs(os.path.dirname(current_shariah_rules_explicit_path) or '.', exist_ok=True) # Ensure dir exists
            with open(current_shariah_rules_explicit_path, "w") as f: json.dump(dummy_rules, f)

        persist_db_path = data.get('persist_db_path_base', "./db_store_api")
        os.makedirs(persist_db_path, exist_ok=True)
        
        # --- Document Processor and Vector Stores ---
        logger.info("Initializing DocumentProcessor...")
        asave_context["doc_processor"] = DocumentProcessor()

        if fas_filepaths:
            logger.info(f"Processing {len(fas_filepaths)} FAS documents for vector store...")
            all_fas_chunks = []
            for fp in fas_filepaths:
                docs = asave_context["doc_processor"].load_pdf(fp)
                chunks = asave_context["doc_processor"].chunk_text(docs)
                all_fas_chunks.extend(chunks)
            if all_fas_chunks:
                fas_db_path = os.path.join(persist_db_path, "fas_db")
                asave_context["fas_vector_store"] = asave_context["doc_processor"].create_vector_store(all_fas_chunks, persist_directory=fas_db_path)
                asave_context["all_fas_vector_store"] = asave_context["fas_vector_store"] # Used by ISCCA
                logger.info(f"FAS Vector Store created/loaded. Path: {fas_db_path}")
        if ss_filepaths:
            logger.info(f"Processing {len(ss_filepaths)} SS documents for vector store...")
            all_ss_chunks = []
            for fp in ss_filepaths:
                docs = asave_context["doc_processor"].load_pdf(fp)
                chunks = asave_context["doc_processor"].chunk_text(docs)
                all_ss_chunks.extend(chunks)
            if all_ss_chunks:
                ss_db_path = os.path.join(persist_db_path, "ss_db")
                asave_context["ss_vector_store"] = asave_context["doc_processor"].create_vector_store(all_ss_chunks, persist_directory=ss_db_path)
                logger.info(f"SS Vector Store created/loaded. Path: {ss_db_path}")

        # --- Initialize Agents ---
        logger.info("Initializing Core Agents (Validation, SRMA)...")
        asave_context["scva_iscca"] = ValidationAgent()
        asave_context["srma"] = ShariahRuleMinerAgent()

        logger.info("Initializing AISGA Variants...")
        asave_context["aisga_variants"]["pro_detailed_conservative"] = SuggestionAgent(
            model_name="gemini-1.5-pro-latest", temperature=0.2,
            system_message="You are a meticulous AAOIFI standards drafter. Prioritize extreme clarity, referencing specific standard sections where possible, and ensure strict Shari'ah compliance. Your suggestions should be conservative and build upon existing text with minimal disruption unless absolutely necessary for compliance or clarity."
        )
        asave_context["aisga_variants"]["pro_detailed_conservative"].agent_type = "AISGA_ProDetailedConservative"

        asave_context["aisga_variants"]["flash_creative_options"] = SuggestionAgent(
            model_name="gemini-1.5-flash-latest", temperature=0.7,
            system_message="You are an innovative AAOIFI standards consultant. Your goal is to provide alternative phrasing and creative solutions for clarity and enhancement, even if they are significant departures from the original text. Always maintain Shari'ah compliance as paramount."
        )
        asave_context["aisga_variants"]["flash_creative_options"].agent_type = "AISGA_FlashCreativeOptions"
        
        logger.info("Initializing Specialized Agents...")
        asave_context["specialized_agents"]["conciseness_agent"] = ConcisenessAgent()

        asave_context["initialized"] = True
        logger.info("ASAVE system and all agent variants initialized successfully.")
        
        logger.info("Initializing Text Reformatter Agent...")
        asave_context["text_reformatter_agent"] = TextReformatterAgent()

        raw_config = {
            "output_format": "markdown",
            "use_llm": True,
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "output_format": "markdown",
            "paginate_output" : True,
        }
        config_parser = ConfigParser(raw_config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        asave_context["text_reformatter_marker"] = converter
        
        return jsonify({
            "status": "success", "message": "ASAVE system initialized with multiple agent variants.",
            "fas_vector_store_status": "Created/Loaded" if asave_context["fas_vector_store"] else "Not Created",
            "ss_vector_store_status": "Created/Loaded" if asave_context["ss_vector_store"] else "Not Created",
            "num_aisga_variants": len(asave_context["aisga_variants"]),
            "num_specialized_agents": len(asave_context["specialized_agents"])
        }), 200

    except ValueError as ve:
        logger.error(f"Initialization ValueError: {ve}", exc_info=True)
        asave_context["initialized"] = False
        return jsonify({"status": "error", "message": f"Agent/Component Initialization Error: {str(ve)}"}), 500
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        asave_context["initialized"] = False
        return jsonify({"status": "error", "message": f"General initialization error: {str(e)}"}), 500

@app.route('/analyze_chunk_enhanced', methods=['POST'])
def analyze_chunk_enhanced_api():
    # This is the NON-STREAMING version that returns all results at once.
    # (Code from the previous response for this endpoint, which uses ThreadPoolExecutor)
    global asave_context, executor
    logger.info("Received /analyze_chunk_enhanced request (non-streaming).")
    # ... (Full implementation as provided in the previous answer)
    if not asave_context["initialized"]:
        return jsonify({"status": "error", "message": "ASAVE system not initialized."}), 400
    if not asave_context["scva_iscca"]:
        return jsonify({"status": "error", "message": "Validation agent (SCVA/ISCCA) not initialized."}), 500

    try:
        data = request.get_json()
        if not data: return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

        target_text = data.get("target_text_chunk")
        fas_context_strings = data.get("fas_context_chunks", [])
        ss_context_strings = data.get("ss_context_chunks", [])
        fas_name = data.get("fas_name_for_display", "Unnamed FAS") # This would be fas_doc_id in new flow
        ambiguity_desc = data.get("identified_ambiguity", "User selected text for review/enhancement.")

        if not target_text: return jsonify({"status": "error", "message": "Missing 'target_text_chunk'."}), 400

        raw_suggestions_from_agents = []
        agent_processing_log = []
        future_to_agent = {}

        # AISGA Variants
        for variant_name, aisga_instance in asave_context["aisga_variants"].items():
            if aisga_instance:
                future = executor.submit(
                    process_suggestion_task, aisga_instance, "generate_clarification",
                    original_text=target_text, identified_ambiguity=ambiguity_desc,
                    fas_context_strings=fas_context_strings, ss_context_strings=ss_context_strings,
                    variant_name_override=variant_name
                )
                future_to_agent[future] = {"type": "AISGA", "name": variant_name}
        # Specialized Agents
        conciseness_agent = asave_context["specialized_agents"].get("conciseness_agent")
        if conciseness_agent:
            future = executor.submit(
                process_suggestion_task, conciseness_agent, "make_concise",
                text_to_make_concise=target_text, shariah_context_strings=ss_context_strings
            )
            future_to_agent[future] = {"type": "Specialized", "name": "ConcisenessAgent"}

        for future in as_completed(future_to_agent):
            agent_info = future_to_agent[future]
            try:
                result = future.result()
                if isinstance(result, dict) and "error" not in result and result.get("proposed_text"):
                    raw_suggestions_from_agents.append({
                        "source_agent_type": agent_info["type"], "source_agent_name": agent_info["name"],
                        "suggestion_payload": result
                    })
                    agent_processing_log.append({"agent_name": agent_info["name"], "type": agent_info["type"], "status": "success", "summary": result.get("proposed_text", "")[:50]+"..."})
                else:
                    agent_processing_log.append({"agent_name": agent_info["name"], "type": agent_info["type"], "status": "failed_or_empty", "details": result})
            except Exception as exc:
                agent_processing_log.append({"agent_name": agent_info["name"], "type": agent_info["type"], "status": "exception_in_future", "error": str(exc)})
        
        validated_suggestions_list = []
        validation_futures = {}
        for raw_sugg_item in raw_suggestions_from_agents:
            suggestion_payload = raw_sugg_item["suggestion_payload"]
            scva_future = executor.submit(
                asave_context["scva_iscca"].validate_shariah_compliance,
                proposed_suggestion_object=suggestion_payload,
                shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"],
                ss_vector_store=asave_context["ss_vector_store"],
                mined_shariah_rules_path=asave_context["mined_shariah_rules_path"] if os.path.exists(asave_context["mined_shariah_rules_path"]) else None
            )
            iscca_future = executor.submit(
                asave_context["scva_iscca"].validate_inter_standard_consistency,
                proposed_suggestion_object=suggestion_payload, fas_name=fas_name,
                all_fas_vector_store=asave_context["all_fas_vector_store"]
            )
            validation_futures[raw_sugg_item] = (scva_future, iscca_future)

        for raw_sugg_item, (scva_future, iscca_future) in validation_futures.items():
            try:
                scva_report = scva_future.result()
                iscca_report = iscca_future.result()
                validated_suggestions_list.append({
                    "source_agent_type": raw_sugg_item["source_agent_type"],
                    "source_agent_name": raw_sugg_item["source_agent_name"],
                    "suggestion_details": raw_sugg_item["suggestion_payload"],
                    "scva_report": scva_report, "iscca_report": iscca_report,
                    "validation_summary_score": f"SCVA: {scva_report.get('overall_status', 'N/A')}, ISCCA: {iscca_report.get('status', 'N/A')}"
                })
            except Exception as exc:
                 logger.error(f"Error collecting validation results for suggestion from {raw_sugg_item['source_agent_name']}: {exc}", exc_info=True)


        final_response = {
            "input_summary": {"target_text_chunk": target_text, "fas_name": fas_name},
            "agent_processing_log": agent_processing_log,
            "validated_suggestions": validated_suggestions_list
        }
        return jsonify({"status": "success", "analysis": final_response}), 200
    except Exception as e:
        logger.error(f"Error during enhanced chunk analysis (non-streaming): {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"General analysis error: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status_api():
    global asave_context
    return jsonify({
        "service_status": "running",
        "asave_initialized": asave_context["initialized"],
        "config": {
            "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
            "upload_folder": app.config['UPLOAD_FOLDER'],
            "explicit_rules_path": asave_context["shariah_rules_explicit_path"],
            "mined_rules_path_default_location": asave_context["mined_shariah_rules_path"]
        },
        "components_loaded": {
            "doc_processor": bool(asave_context["doc_processor"]),
            "fas_vector_store": bool(asave_context["fas_vector_store"]),
            "ss_vector_store": bool(asave_context["ss_vector_store"]),
            "num_aisga_variants": len(asave_context["aisga_variants"]),
            "num_specialized_agents": len(asave_context["specialized_agents"]),
            "scva_iscca": bool(asave_context["scva_iscca"]),
            "srma": bool(asave_context["srma"]),
            "text_reformatter_agent": bool(asave_context["text_reformatter_agent"]),
            "text_reformatter_marker": bool(asave_context["text_reformatter_marker"])
        }
    })

@app.route('/mine_shariah_rules', methods=['POST'])
def mine_shariah_rules_api():
    global asave_context
    # ... (Code from previous answer, no changes needed here for SSE feature) ...
    logger.info("Received /mine_shariah_rules request.")
    if not asave_context["initialized"] or not asave_context["srma"] or not asave_context["doc_processor"]:
        return jsonify({"status": "error", "message": "ASAVE system or SRMA components not initialized."}), 400
    try:
        data = request.form; uploaded_files = request.files.getlist('ss_files_for_srma')
        ss_metadata_list = []
        for i, file in enumerate(uploaded_files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename); filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(filepath)
                full_name = data.get(f'ss_files_for_srma_{i}_fullname', f"Unnamed Standard {os.path.splitext(filename)[0]}")
                short_code = data.get(f'ss_files_for_srma_{i}_shortcode', f"USS{i+1}")
                ss_metadata_list.append({"filepath": filepath, "standard_name_full": full_name, "standard_short_code": short_code.upper()})
        if not ss_metadata_list: return jsonify({"status": "error", "message": "No valid SS files or metadata provided for SRMA."}), 400
        output_dir = data.get('output_directory', "output_srma_api"); os.makedirs(output_dir, exist_ok=True)
        combined_rules_path = asave_context["srma"].mine_rules_from_document_list(
            ss_documents_with_metadata=ss_metadata_list, doc_processor_instance=asave_context["doc_processor"], base_output_dir=output_dir)
        if combined_rules_path and os.path.exists(combined_rules_path):
            asave_context["mined_shariah_rules_path"] = combined_rules_path
            return jsonify({"status": "success", "message": "SRMA processing complete.", "output_file_path": combined_rules_path, "num_files_processed": len(ss_metadata_list)}), 200
        else: return jsonify({"status": "error", "message": "SRMA failed to produce an output file."}), 500
    except Exception as e:
        logger.error(f"Error during SRMA execution: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"General SRMA error: {str(e)}"}), 500

# --- STREAMING API Endpoint ---

@app.route('/get_assistance_stream', methods=['POST'])
def get_assistance_stream_api():
    global asave_context, executor
    logger.info("Received /get_assistance_stream request.")

    # Basic checks
    if not asave_context["initialized"]:
        def error_stream_init(): yield stream_event({"event_type": "error", "step": "INITIALIZATION", "message": "ASAVE system not initialized."})
        return Response(stream_with_context(error_stream_init()), mimetype='text/event-stream', status=400)
    if not asave_context["scva_iscca"]:
        def error_stream_val(): yield stream_event({"event_type": "error", "step": "INITIALIZATION", "message": "Validation agent not initialized."})
        return Response(stream_with_context(error_stream_val()), mimetype='text/event-stream', status=400)

    try:
        data = request.get_json()
        if not data:
            def error_stream_payload(): yield stream_event({"event_type": "error", "step": "REQUEST_VALIDATION", "message": "Invalid JSON payload."})
            return Response(stream_with_context(error_stream_payload()), mimetype='text/event-stream', status=400)

        selected_text = data.get("selected_text_from_fas")
        fas_doc_id = data.get("fas_document_id") # Used for ISCCA and potentially FAS context retrieval
        # Optional user-provided keywords for SS context to guide retrieval
        # user_ss_keywords = data.get("user_provided_ss_context_keywords", []) 

        if not selected_text or not fas_doc_id:
            def error_stream_missing_data(): yield stream_event({"event_type": "error", "step": "REQUEST_VALIDATION", "message": "Missing 'selected_text_from_fas' or 'fas_document_id'."})
            return Response(stream_with_context(error_stream_missing_data()), mimetype='text/event-stream', status=400)

    except Exception as e_req:
        logger.error(f"Error processing request for stream: {e_req}", exc_info=True)
        def error_stream_req_proc(): yield stream_event({"event_type": "error", "step": "REQUEST_PROCESSING", "message": f"Error processing request: {str(e_req)}."})
        return Response(stream_with_context(error_stream_req_proc()), mimetype='text/event-stream', status=400)

    # --- Generator function for SSE ---
    def generate_analysis_stream_content():
        try:
            yield stream_event({"event_type": "system_log", "step": "STREAM_START", "message": "Analysis stream initiated..."})
            time.sleep(0.05) # Ensure first message gets through

            # Step 1: Retrieve Relevant SS Context (RAG for SS)
            yield stream_event({"event_type": "progress", "step_code": "SS_CTX_START", "message": "🔍 Retrieving Shari'ah context..."})
            ss_context_strings = []
            if asave_context["ss_vector_store"]:
                try:
                    query_for_ss = selected_text # Could be enhanced with keywords
                    retriever = asave_context["ss_vector_store"].as_retriever(search_kwargs={"k": 5})
                    relevant_ss_docs = retriever.get_relevant_documents(query_for_ss)
                    ss_context_strings = [doc.page_content for doc in relevant_ss_docs]
                    yield stream_event({"event_type": "progress", "step_code": "SS_CTX_DONE", "message": f"Retrieved {len(ss_context_strings)} SS context snippets.", "payload": {"count": len(ss_context_strings)}})
                except Exception as e:
                    logger.error(f"Stream: Error retrieving SS context: {e}")
                    yield stream_event({"event_type": "warning", "step_code": "SS_CTX_ERROR", "message": f"SS context retrieval failed: {str(e)}"})
            else:
                yield stream_event({"event_type": "warning", "step_code": "SS_CTX_SKIP", "message": "SS Vector Store not available."})
            time.sleep(0.05)

            # Step 2: Retrieve Relevant FAS Context (Self-Context from the current FAS)
            # This assumes fas_vector_store can be filtered or is specific to fas_doc_id
            yield stream_event({"event_type": "progress", "step_code": "FAS_CTX_START", "message": "📑 Retrieving local FAS context..."})
            fas_context_strings = []
            if asave_context["fas_vector_store"]: # Assuming this store might contain multiple FAS
                try:
                    # Conceptual: filter by fas_doc_id if metadata 'source' contains it
                    # retriever = asave_context["fas_vector_store"].as_retriever(search_kwargs={"k": 3, "filter": {"source": fas_doc_id}})
                    # For now, query generally and assume client might have sent some or we use generic context.
                    retriever = asave_context["fas_vector_store"].as_retriever(search_kwargs={"k": 3})
                    relevant_fas_docs = retriever.get_relevant_documents(selected_text)
                    fas_context_strings = [doc.page_content for doc in relevant_fas_docs]
                    yield stream_event({"event_type": "progress", "step_code": "FAS_CTX_DONE", "message": f"Retrieved {len(fas_context_strings)} local FAS context snippets.", "payload": {"count": len(fas_context_strings)}})
                except Exception as e:
                    logger.error(f"Stream: Error retrieving FAS context: {e}")
                    yield stream_event({"event_type": "warning", "step_code": "FAS_CTX_ERROR", "message": f"FAS context retrieval failed: {str(e)}"})
            else:
                yield stream_event({"event_type": "warning", "step_code": "FAS_CTX_SKIP", "message": "FAS Vector Store not available."})
            time.sleep(0.05)

            # Step 3: Invoke Suggestion Agents in Parallel
            suggestion_futures = {}
            # AISGA Variants
            for variant_name, aisga_instance in asave_context["aisga_variants"].items():
                if aisga_instance:
                    yield stream_event({"event_type": "progress", "step_code": f"AISGA_START_{variant_name.upper()}", "agent_name": variant_name, "message": f"AISGA '{variant_name}' drafting..."})
                    future = executor.submit(process_suggestion_task, aisga_instance, "generate_clarification",
                                             original_text=selected_text, identified_ambiguity="User selection from FAS for review.",
                                             fas_context_strings=fas_context_strings, ss_context_strings=ss_context_strings,
                                             variant_name_override=variant_name)
                    suggestion_futures[future] = {"type": "AISGA", "name": variant_name}
            # Specialized Agents
            conciseness_agent = asave_context["specialized_agents"].get("conciseness_agent")
            if conciseness_agent:
                yield stream_event({"event_type": "progress", "step_code": "CONCISENESS_START", "agent_name": "ConcisenessAgent", "message": "Conciseness Agent working..."})
                future = executor.submit(process_suggestion_task, conciseness_agent, "make_concise",
                                         text_to_make_concise=selected_text, shariah_context_strings=ss_context_strings)
                suggestion_futures[future] = {"type": "Specialized", "name": "ConcisenessAgent"}

            raw_suggestions_for_validation = []
            for future in as_completed(suggestion_futures):
                agent_info = suggestion_futures[future]
                try:
                    result = future.result()
                    if isinstance(result, dict) and "error" not in result and result.get("proposed_text"):
                        raw_suggestions_for_validation.append({"source_agent_info": agent_info, "suggestion_payload": result})
                        yield stream_event({"event_type": "agent_suggestion_generated", "step_code": f"SUGGESTION_GEN_{agent_info['name'].upper()}",
                                            "agent_name": agent_info['name'], "agent_type": agent_info['type'],
                                            "message": f"Suggestion received from '{agent_info['name']}'.",
                                            "payload": {"suggestion_summary": result.get("proposed_text", "")[:70]+"..." } # Send summary
                                        })
                    else:
                        yield stream_event({"event_type": "warning", "step_code": f"SUGGESTION_FAIL_{agent_info['name'].upper()}", "agent_name": agent_info['name'], "message": f"Agent '{agent_info['name']}' failed or returned empty.", "payload": result})
                except Exception as e_sugg:
                    yield stream_event({"event_type": "error", "step_code": f"SUGGESTION_EXC_{agent_info['name'].upper()}", "agent_name": agent_info['name'], "message": f"Error with agent '{agent_info['name']}': {str(e_sugg)}"})
                time.sleep(0.05)


            # Step 4: Validate Each Suggestion in Parallel
            if not raw_suggestions_for_validation:
                 yield stream_event({"event_type": "system_log", "step_code": "VALIDATION_SKIP_NO_SUGG", "message": "No suggestions were generated for validation."})
            
            validation_futures_map = {} # Map (scva_future, iscca_future) tuple to original suggestion info
            for i, item in enumerate(raw_suggestions_for_validation):
                sugg_payload = item["suggestion_payload"]
                agent_info = item["source_agent_info"]
                yield stream_event({"event_type": "progress", "step_code": f"VALIDATION_START_{agent_info['name'].upper()}_Sugg{i}", "agent_name": agent_info['name'], "message": f"Validating suggestion from '{agent_info['name']}'..."})
                
                scva_future = executor.submit(asave_context["scva_iscca"].validate_shariah_compliance,
                                              proposed_suggestion_object=sugg_payload, shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"],
                                              ss_vector_store=asave_context["ss_vector_store"],
                                              mined_shariah_rules_path=asave_context["mined_shariah_rules_path"] if os.path.exists(asave_context["mined_shariah_rules_path"]) else None)
                iscca_future = executor.submit(asave_context["scva_iscca"].validate_inter_standard_consistency,
                                               proposed_suggestion_object=sugg_payload, fas_name=fas_doc_id,
                                               all_fas_vector_store=asave_context["all_fas_vector_store"])
                validation_futures_map[(scva_future, iscca_future)] = item # Store original item info

            total_validated_count = 0
            for (scva_future, iscca_future), original_item_info in validation_futures_map.items():
                agent_name = original_item_info["source_agent_info"]["name"]
                try:
                    scva_report = scva_future.result()
                    iscca_report = iscca_future.result()
                    validated_suggestion_package = {
                        "source_agent_type": original_item_info["source_agent_info"]["type"],
                        "source_agent_name": agent_name,
                        "suggestion_details": original_item_info["suggestion_payload"],
                        "scva_report": scva_report,
                        "iscca_report": iscca_report,
                        "validation_summary_score": f"SCVA: {scva_report.get('overall_status', 'N/A')}, ISCCA: {iscca_report.get('status', 'N/A')}"
                    }
                    total_validated_count += 1
                    yield stream_event({"event_type": "validated_suggestion_package", "step_code": f"VALIDATION_DONE_{agent_name.upper()}",
                                        "agent_name": agent_name, "message": f"Validation complete for '{agent_name}'.",
                                        "payload": validated_suggestion_package # Send the full validated package
                                    })
                except Exception as e_val_collect:
                    yield stream_event({"event_type": "error", "step_code": f"VALIDATION_EXC_{agent_name.upper()}", "agent_name": agent_name, "message": f"Error collecting validation for '{agent_name}': {str(e_val_collect)}"})
                time.sleep(0.05)

            yield stream_event({"event_type": "system_log", "step_code": "STREAM_END", "message": "Analysis stream finished.", "payload": {"total_validated_suggestions": total_validated_count}})

        except Exception as e_stream_main:
            logger.error(f"Critical error within stream generation: {e_stream_main}", exc_info=True)
            yield stream_event({"event_type": "fatal_error", "step_code": "STREAM_GENERATOR_ERROR", "message": f"Stream failed: {str(e_stream_main)}"})

    # This initiates the generator and returns a streaming response
    return Response(stream_with_context(generate_analysis_stream_content()), mimetype='text/event-stream')


@app.route('/extract_text_from_pdf', methods=['POST'])
def extract_text_from_pdf_api():
    logger.info("Received /extract_text_from_pdf request (with AI reformatting).")
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No 'pdf_file' part in the request."}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400

    reformatter_agent = asave_context.get("text_reformatter_agent")
    if not reformatter_agent:
        logger.error("TextReformatterAgent not initialized. Cannot perform AI reformatting.")
        # Fallback to raw extraction or return an error
        # For now, let's proceed with raw if formatter is missing, but log heavily.
        # A better approach might be to make it configurable via a request param.
    
    # Determine if AI reformatting should be applied (e.g., via a query parameter)
    apply_ai_reformatting = request.args.get('reformat', 'true').lower() == 'true'


    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            reformatted_pages_markdown = []
            raw_extracted_pages_for_fallback = [] # Store raw text in case AI fails

            metadata = {
                "filename": filename,
                "page_count": doc.page_count,
                "metadata": doc.metadata,
                "ai_reformatting_applied": reformatter_agent is not None and apply_ai_reformatting
            }

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                # Get raw text first - this is still the most reliable way to get *all* text content
                # Using simple text extraction as input for the LLM
                raw_page_text = page.get_text("text", sort=True).strip()
                raw_extracted_pages_for_fallback.append({
                    "page_number": page_num + 1,
                    "content_type": "raw_text",
                    "content": raw_page_text
                })

                if reformatter_agent and apply_ai_reformatting:
                    logger.info(f"AI Reformatting page {page_num + 1}/{doc.page_count} for {filename}...")
                    # This call can be slow, consider for background processing for large docs if UX demands it
                    reformat_result = reformatter_agent.reformat_to_markdown(raw_page_text, page_number=page_num + 1)
                    
                    if reformat_result["status"] == "success":
                        reformatted_pages_markdown.append({
                            "page_number": page_num + 1,
                            "content_type": "markdown",
                            "content": reformat_result["markdown_content"],
                            "reformat_notes": reformat_result.get("notes")
                        })
                    else:
                        logger.warning(f"AI reformatting failed for page {page_num + 1}: {reformat_result.get('error_message')}. Using raw text.")
                        reformatted_pages_markdown.append({
                            "page_number": page_num + 1,
                            "content_type": "markdown_failed_fallback_raw_text", # Indicate fallback
                            "content": raw_page_text, # Fallback to raw text formatted as basic Markdown paragraph
                            "reformat_notes": f"AI reformatting failed: {reformat_result.get('error_message')}. Raw text shown."
                        })
                else: # No reformatter agent or reformatting disabled
                    reformatted_pages_markdown.append({
                        "page_number": page_num + 1,
                        "content_type": "raw_text_as_markdown_paragraph", # Raw text, but client will treat as MD
                        "content": raw_page_text, # Treat raw text as a single block of Markdown
                        "reformat_notes": "AI reformatting not applied." if apply_ai_reformatting else "AI reformatting disabled by request."
                    })
            
            doc.close()
            logger.info(f"Text processing complete for {filename}. AI reformatting applied: {metadata['ai_reformatting_applied']}")
            
            # The `pages` array will now contain Markdown content (or raw text if AI failed/skipped)
            return jsonify({
                "status": "success",
                "message": "Text processed successfully.",
                "document_info": metadata,
                "pages": reformatted_pages_markdown # This now contains Markdown
            }), 200

        except Exception as e:
            logger.error(f"Error processing PDF {file.filename} for text extraction/reformatting: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Failed to process PDF: {str(e)}"}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type, PDF required."}), 400


@app.route('/extract_text_from_pdf_file_marker', methods=['POST'])
def extract_text_from_pdf_file_api():
    """
    Accepts a PDF file upload, extracts text and images using the marker pipeline, and returns the result as JSON.
    """
    logger.info("Received /extract_text_from_pdf_file request.")
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No 'pdf_file' part in the request."}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type, PDF required."}), 400

    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        converter = asave_context.get("text_reformatter_marker")
        rendered = converter(temp_path)
        text, _, images = text_from_rendered(rendered)

        image_info = []
        if images:
            for path, image in images.items():
                try:
                    os.makedirs('./output', exist_ok=True)
                    image_save_path = os.path.join('./output', path)
                    image.save(image_save_path, image.format)
                    image_info.append({"image_path": path, "saved_to": image_save_path, "format": image.format})
                except Exception as e:
                    image_info.append({"image_path": path, "error": str(e)})
        else:
            image_info = "No images found in the PDF."

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as cleanup_err:
            logger.warning(f"Could not remove temp file {temp_path}: {cleanup_err}")

        return jsonify({
            "status": "success",
            "message": "Extracted text and images from uploaded PDF.",
            "filename": filename,
            "extracted_text": text,
            "images": image_info
        }), 200

    except Exception as e:
        logger.error(f"Error extracting text/images from PDF: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to process PDF: {str(e)}"}), 500

def process_agent_task(agent_instance, method_name: str, task_description: str, **kwargs) -> dict:
    """
    Helper function to call agent methods, designed for use with ThreadPoolExecutor.
    It standardizes the output to include status, task description, agent type,
    and either the result or an error message.

    Args:
        agent_instance: The instance of the agent to call.
        method_name (str): The name of the method to call on the agent instance.
        task_description (str): A descriptive string for this task (for logging/tracking).
        **kwargs: Keyword arguments to pass to the agent's method.

    Returns:
        dict: A dictionary containing:
              - "status": "success" or "error"
              - "task": The provided task_description
              - "agent_type": The class name of the agent (or a defined agent_type attribute)
              - "agent_name_variant": The specific variant name if applicable (e.g., from AISGA variants)
              - "result": The result from the agent method if successful.
              - "error": The error message if an error occurred.
              - "prompt_details_actual": If the result itself contains this key (from AISGA etc.)
                                         it will be part of the 'result' dict.
    """
    # Try to get a more specific agent type/name for logging
    agent_class_name = type(agent_instance).__name__
    # Some of your agents (like AISGA variants or specialized agents) might have an 'agent_type' attribute
    # or the variant_name might be passed in kwargs for AISGA.
    specific_agent_name = kwargs.get("variant_name_override", None) # Check if explicitly passed
    if not specific_agent_name: # If not overridden, try to get from instance or use class name
        specific_agent_name = getattr(agent_instance, 'agent_type', agent_class_name)
    
    full_task_id = f"{task_description} (Agent: {specific_agent_name}, Method: {method_name})"
    logger.info(f"EXECUTING Task: {full_task_id} with args: {list(kwargs.keys())}")

    response_payload = {
        "status": "error", # Default to error
        "task": task_description,
        "agent_type": agent_class_name, # e.g., SuggestionAgent, ValidationAgent
        "agent_name_variant": specific_agent_name, # e.g., AISGA_ProDetailedConservative
        "result": None,
        "error": "Unknown error occurred."
    }

    try:
        if not agent_instance:
            response_payload["error"] = "Agent instance is None."
            logger.error(f"FAILED Task: {full_task_id}. Reason: Agent instance is None.")
            return response_payload

        if not hasattr(agent_instance, method_name):
            response_payload["error"] = f"Method '{method_name}' not found in agent '{agent_class_name}'."
            logger.error(f"FAILED Task: {full_task_id}. Reason: Method not found.")
            return response_payload

        method_to_call = getattr(agent_instance, method_name)
        
        # Inspect the signature of the method to be called
        sig = inspect.signature(method_to_call)
        method_params = sig.parameters

        # Prepare arguments for the method call, only passing what it accepts
        call_kwargs = {}
        for param_name, param_obj in method_params.items():
            if param_name == 'self': # Skip 'self'
                continue
            if param_name in kwargs:
                call_kwargs[param_name] = kwargs[param_name]
            # If param has a default value and not in kwargs, it will use its default
            # If param has no default and not in kwargs, it will raise TypeError (which is good)

        # If 'variant_name_override' was in kwargs but not directly a param of the method,
        # some agent methods (like AISGA's generate_clarification) might accept 'variant_name'.
        # The AISGA `generate_clarification` was modified to accept `variant_name`.
        # This logic is now handled more directly in the AISGA method itself if `variant_name` is passed.
        # The `variant_name_override` in kwargs is primarily for identifying the task source.
        
        agent_result = method_to_call(**call_kwargs)

        # Check if the agent_result itself indicates an error (some agents return dicts with "error" key)
        if isinstance(agent_result, dict) and "error" in agent_result:
            response_payload["error"] = agent_result["error"]
            # If the agent's error payload also contains 'llm_response_received' or 'prompt_details_actual', include them for debugging
            if "llm_response_received" in agent_result:
                response_payload["llm_response_on_agent_error"] = agent_result["llm_response_received"]
            if "prompt_details_actual" in agent_result: # If agent includes prompt details even on error
                 response_payload["prompt_details_actual_on_agent_error"] = agent_result["prompt_details_actual"]
            logger.warning(f"Task PARTIALLY FAILED (agent reported error): {full_task_id}. Agent Error: {agent_result['error']}")
        else:
            response_payload["status"] = "success"
            response_payload["result"] = agent_result
            response_payload["error"] = None # Clear default error
            logger.info(f"COMPLETED Task: {full_task_id} successfully.")

    except TypeError as te: # Catches errors from calling method with wrong args
        error_msg = f"TypeError calling {method_name} on {specific_agent_name}: {te}. Check arguments."
        logger.error(f"FAILED Task: {full_task_id}. Reason: {error_msg}", exc_info=True)
        response_payload["error"] = error_msg
    except Exception as e:
        error_msg = f"General exception calling {method_name} on {specific_agent_name}: {e}"
        logger.error(f"FAILED Task: {full_task_id}. Reason: {error_msg}", exc_info=True)
        response_payload["error"] = error_msg

    return response_payload


# --- NEW STREAMING ENDPOINT for Shari'ah Contract Helper ---
@app.route('/validate_contract_terms_stream', methods=['POST'])
def validate_contract_terms_stream_api():
    global asave_context, executor # Ensure executor is accessible
    logger.info("Received /validate_contract_terms_stream request.")

    # --- Initial Request Validation and Component Checks ---
    if not asave_context.get("initialized"):
        def error_stream_uninit(): yield stream_event({"event_type": "fatal_error", "step_code": "SYSTEM_UNINITIALIZED", "message": "ASAVE system core components not initialized. Please call /initialize first."})
        return Response(stream_with_context(error_stream_uninit()), mimetype='text/event-stream', status=503) # Service Unavailable

    validation_agent = asave_context.get("scva_iscca")
    aisga_agents = asave_context.get("aisga_variants")
    if not validation_agent or not aisga_agents or not isinstance(aisga_agents, dict) or not aisga_agents:
        def error_stream_agents_missing(): yield stream_event({"event_type": "fatal_error", "step_code": "AGENTS_MISSING", "message": "Essential AI agents (Validation or Suggestion) are not available. Please check server initialization."})
        return Response(stream_with_context(error_stream_agents_missing()), mimetype='text/event-stream', status=503)

    try:
        data = request.get_json()
        if not data:
            def error_stream_payload(): yield stream_event({"event_type": "fatal_error", "step_code": "INVALID_PAYLOAD", "message": "Invalid JSON payload."})
            return Response(stream_with_context(error_stream_payload()), mimetype='text/event-stream', status=400) # Bad Request

        contract_type_input = data.get("contract_type", "General Contract")
        client_clauses_raw = data.get("client_clauses") # Expected: [{"clause_id": "c1", "text": "..."}, ...]
        overall_contract_context_str = data.get("overall_contract_context", "")

        if not client_clauses_raw or not isinstance(client_clauses_raw, list) or \
           not all(isinstance(c, dict) and "text" in c and isinstance(c["text"], str) for c in client_clauses_raw):
            def error_stream_clauses_invalid(): yield stream_event({"event_type": "fatal_error", "step_code": "INVALID_CLAUSES", "message": "Missing or invalid 'client_clauses' array. Each clause must be an object with a 'text' field."})
            return Response(stream_with_context(error_stream_clauses_invalid()), mimetype='text/event-stream', status=400)
        
        # Assign unique IDs if not provided, and filter empty clauses
        client_clauses = []
        for i, c_raw in enumerate(client_clauses_raw):
            if c_raw["text"].strip(): # Only process non-empty clauses
                client_clauses.append({
                    "clause_id": c_raw.get("clause_id") or f"client_clause_{i+1}_{time.time_ns()}",
                    "text": c_raw["text"]
                })
        
        if not client_clauses:
            def error_stream_no_valid_clauses(): yield stream_event({"event_type": "fatal_error", "step_code": "NO_VALID_CLAUSES", "message": "No valid (non-empty) client clauses provided for analysis."})
            return Response(stream_with_context(error_stream_no_valid_clauses()), mimetype='text/event-stream', status=400)


    except Exception as e_req: # Catch errors during request parsing
        logger.error(f"Error processing request for contract validation stream: {e_req}", exc_info=True)
        def error_stream_req_proc(): yield stream_event({"event_type": "fatal_error", "step_code": "REQUEST_PROCESSING_ERROR", "message": f"Error processing request data: {str(e_req)}."})
        return Response(stream_with_context(error_stream_req_proc()), mimetype='text/event-stream', status=400)

    # --- Generator function for SSE ---
    def generate_contract_validation_stream():
        try:
            yield stream_event({"event_type": "system_log", "step_code": "CONTRACT_STREAM_START", 
                                "message": f"Contract validation initiated for type '{contract_type_input}' with {len(client_clauses)} clauses."})
            time.sleep(0.05) # Allow frontend to register start

            # --- Step 1: Load Relevant Shari'ah Context (SS Vector Store) once for all clauses ---
            yield stream_event({"event_type": "progress", "step_code": "SS_CTX_LOAD_START", 
                                "message": f"🔍 Loading Shari'ah Standards context for '{contract_type_input}'..."})
            ss_context_for_all_clauses = []
            if asave_context.get("ss_vector_store"):
                try:
                    combined_query_text = f"{contract_type_input} {overall_contract_context_str} " + " ".join([c["text"][:150] for c in client_clauses])
                    retriever = asave_context["ss_vector_store"].as_retriever(search_kwargs={"k": 7}) # Get a good number of general SS contexts
                    relevant_ss_docs = retriever.get_relevant_documents(combined_query_text)
                    ss_context_for_all_clauses = [doc.page_content for doc in relevant_ss_docs]
                    yield stream_event({"event_type": "progress", "step_code": "SS_CTX_LOAD_DONE", 
                                        "message": f"Retrieved {len(ss_context_for_all_clauses)} SS context snippets.", 
                                        "payload": {"count": len(ss_context_for_all_clauses)}})
                except Exception as e_ss_ctx:
                    logger.error(f"Stream: Error retrieving SS context for contract: {e_ss_ctx}")
                    yield stream_event({"event_type": "warning", "step_code": "SS_CTX_LOAD_ERROR", 
                                        "message": f"SS context retrieval failed: {str(e_ss_ctx)}"})
            else:
                yield stream_event({"event_type": "warning", "step_code": "SS_CTX_LOAD_SKIP", 
                                    "message": "SS Vector Store not available for context."})
            time.sleep(0.05)

            # --- Process each client clause ---
            for clause_item in client_clauses:
                clause_id = clause_item["clause_id"]
                clause_text = clause_item["text"]

                yield stream_event({"event_type": "clause_processing_start", 
                                    "payload": {"clause_id": clause_id, "original_text": clause_text, 
                                                "message": f"Analyzing clause ({clause_id}): '{clause_text[:60]}...'"}})
                
                # --- SCVA Validation for the client's original clause (using batched method) ---
                scva_task_desc = f"SCVA_batch_for_client_clause_{clause_id}"
                # Submit SCVA task to executor - this is a single LLM call for all rules
                scva_future = executor.submit(
                    process_agent_task, # Your helper
                    agent_instance=validation_agent, 
                    method_name="validate_shariah_compliance_batched", 
                    task_description=scva_task_desc,
                    # Kwargs for validate_shariah_compliance_batched:
                    proposed_suggestion_object={"proposed_text": clause_text, "shariah_notes": f"Client proposed clause for {contract_type_input}. Context: {overall_contract_context_str}"},
                    shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"],
                    ss_vector_store=asave_context["ss_vector_store"], # For semantic check within batched method
                    mined_shariah_rules_path=asave_context["mined_shariah_rules_path"] if os.path.exists(asave_context["mined_shariah_rules_path"]) else None,
                    contract_type=contract_type_input
                )
                
                scva_result_wrapper = scva_future.result() # Wait for SCVA for this clause
                client_clause_scva_report = None
                if scva_result_wrapper["status"] == "success":
                    client_clause_scva_report = scva_result_wrapper["result"]
                    yield stream_event({"event_type": "clause_validation_result", 
                                        "payload": {"clause_id": clause_id, "original_text": clause_text, 
                                                    "scva_report": client_clause_scva_report }})
                else:
                    yield stream_event({"event_type": "error", "step_code": f"SCVA_FAIL_{clause_id}", 
                                        "message": f"SCVA validation failed for clause {clause_id}: {scva_result_wrapper['error']}", 
                                        "payload": {"clause_id": clause_id, "error": scva_result_wrapper['error']}})
                time.sleep(0.05)

                # --- AISGA Suggestions if needed or for enhancement ---
                # Determine if suggestions are warranted based on SCVA report
                needs_ai_suggestion = True # Default, or refine this logic
                if client_clause_scva_report:
                    scva_status = client_clause_scva_report.get("overall_status", "").lower()
                    if scva_status.startswith("compliant"):
                        # Still proceed to get enhancement suggestions
                        yield stream_event({"event_type": "system_log", "step_code": f"AISGA_ENHANCE_{clause_id}", 
                                            "message": f"Clause {clause_id} initially compliant. Requesting AISGA enhancements."})
                    else: # Non-Compliant or Needs Expert Review
                         yield stream_event({"event_type": "system_log", "step_code": f"AISGA_CORRECT_{clause_id}", 
                                            "message": f"Clause {clause_id} is '{scva_status}'. Requesting AISGA corrections/alternatives."})
                
                if needs_ai_suggestion:
                    aisga_suggestion_futures = {}
                    ambiguity_for_aisga = (
                        f"Client proposed clause for a {contract_type_input} agreement. "
                        f"Initial Shari'ah validation by SCVA: Status='{client_clause_scva_report.get('overall_status', 'N/A') if client_clause_scva_report else 'Not available'}'; "
                        f"Reason='{client_clause_scva_report.get('summary_explanation', 'N/A') if client_clause_scva_report else 'N/A'}'. "
                        f"Concerns found: {len(client_clause_scva_report.get('explicit_rule_batch_assessment', {}).get('identified_issues', []))} issues from rule set. "
                        f"Please propose compliant alternatives or enhancements to the original clause text below, considering the provided Shari'ah Standards context."
                    )

                    for variant_name, aisga_instance in aisga_agents.items():
                        yield stream_event({"event_type": "progress", "step_code": f"AISGA_START_{variant_name.upper()}_{clause_id}", 
                                            "agent_name": variant_name, "message": f"AISGA Variant '{variant_name}' drafting for clause {clause_id}..."})
                        future = executor.submit(
                            process_agent_task, aisga_instance, "generate_clarification",
                            task_description=f"AISGA_{variant_name}_for_{clause_id}",
                            original_text=clause_text, identified_ambiguity=ambiguity_for_aisga,
                            fas_context_strings=[], # No external FAS standard context for client clauses
                            ss_context_strings=ss_context_for_all_clauses, # Pass the general SS context
                            variant_name_override=variant_name
                        )
                        aisga_suggestion_futures[future] = {"type": "AISGA", "name": variant_name, "clause_id": clause_id}
                    
                    for future in as_completed(aisga_suggestion_futures):
                        agent_info = aisga_suggestion_futures[future]
                        aisga_sugg_wrapper = future.result()

                        if aisga_sugg_wrapper["status"] == "success":
                            ai_suggestion_payload = aisga_sugg_wrapper["result"] # This is SuggestionAgent's output
                            
                            yield stream_event({"event_type": "progress", "step_code": f"AISGA_SUGG_VALIDATION_START_{agent_info['name']}_{clause_id}", 
                                                "message": f"Validating AISGA ({agent_info['name']}) suggestion for clause {clause_id}..."})
                            
                            # Validate this AI-generated suggestion using SCVA (batched method)
                            scva_aisugg_task_desc = f"SCVA_batch_for_AISuggestion_{agent_info['name']}_{clause_id}"
                            scva_aisugg_future = executor.submit(
                                process_agent_task, validation_agent, "validate_shariah_compliance_batched",
                                task_description=scva_aisugg_task_desc,
                                proposed_suggestion_object=ai_suggestion_payload, # The AI's own suggestion
                                shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"],
                                ss_vector_store=asave_context["ss_vector_store"],
                                mined_shariah_rules_path=asave_context["mined_shariah_rules_path"] if os.path.exists(asave_context["mined_shariah_rules_path"]) else None,
                                contract_type=contract_type_input
                            )
                            scva_aisugg_result_wrapper = scva_aisugg_future.result()
                            ai_suggestion_scva_report = scva_aisugg_result_wrapper.get("result") if scva_aisugg_result_wrapper["status"] == "success" else None

                            # ISCCA for AI's suggestion (optional here, can be added)
                            # iscca_aisugg_report = ...

                            packaged_ai_suggestion_with_validation = {
                                "clause_id": clause_id, # Link back to original client clause
                                "source_agent_type": agent_info["type"],
                                "source_agent_name": agent_info["name"],
                                "suggestion_details": ai_suggestion_payload,
                                "scva_report_on_ai_suggestion": ai_suggestion_scva_report,
                                # "iscca_report_on_ai_suggestion": iscca_aisugg_report, # If ISCCA is run
                                "validation_summary_score": f"AI Sugg. SCVA: {ai_suggestion_scva_report.get('overall_status', 'N/A') if ai_suggestion_scva_report else 'Validation Error'}"
                            }
                            yield stream_event({"event_type": "clause_ai_suggestion_generated", "payload": packaged_ai_suggestion_with_validation})
                        else:
                            yield stream_event({"event_type": "error", "step_code": f"AISGA_FAIL_{agent_info['name']}_{clause_id}", 
                                                "message": f"AISGA variant '{agent_info['name']}' failed for clause {clause_id}: {aisga_sugg_wrapper['error']}", 
                                                "payload": {"clause_id": clause_id, "agent_name": agent_info['name'], "error": aisga_sugg_wrapper['error']}})
                        time.sleep(0.05)
                
                yield stream_event({"event_type": "clause_processing_end", 
                                    "payload": {"clause_id": clause_id, "message": f"Finished analysis for clause ({clause_id})."}})
                time.sleep(0.1)

            yield stream_event({"event_type": "system_log", "step_code": "CONTRACT_STREAM_END", 
                                "message": "Contract terms validation stream finished successfully."})

        except Exception as e_stream_main:
            logger.error(f"Critical error within contract validation stream generator: {e_stream_main}", exc_info=True)
            yield stream_event({"event_type": "fatal_error", "step_code": "STREAM_GENERATOR_ERROR", 
                                "message": f"Stream generation failed catastrophically: {str(e_stream_main)}"})

    return Response(stream_with_context(generate_contract_validation_stream()), mimetype='text/event-stream')


@app.route('/review_full_contract_stream', methods=['POST'])
def review_full_contract_stream_api():
    global asave_context, executor
    logger.info("Received /review_full_contract_stream request.")

    # --- Initial Checks ---
    if not asave_context.get("initialized") or not asave_context.get("scva_iscca"): # scva_iscca is our ValidationAgent
        def error_stream_init(): yield stream_event({"event_type": "fatal_error", "message": "ASAVE system or ValidationAgent not initialized."})
        return Response(stream_with_context(error_stream_init()), mimetype='text/event-stream', status=503)

    try:
        data = request.get_json()
        if not data: # ... return error stream ...
            def error_stream_payload(): yield stream_event({"event_type": "fatal_error", "message": "Invalid JSON payload."})
            return Response(stream_with_context(error_stream_payload()), mimetype='text/event-stream', status=400)

        full_contract_text = data.get("full_contract_text")
        contract_type = data.get("contract_type", "General Contract")

        if not full_contract_text or not isinstance(full_contract_text, str) or not full_contract_text.strip():
            def error_stream_text(): yield stream_event({"event_type": "fatal_error", "message": "Missing or empty 'full_contract_text'."})
            return Response(stream_with_context(error_stream_text()), mimetype='text/event-stream', status=400)
    except Exception as e_req: # ... return error stream ...
        def error_stream_req_proc(): yield stream_event({"event_type": "fatal_error", "message": f"Error processing request: {str(e_req)}."})
        return Response(stream_with_context(error_stream_req_proc()), mimetype='text/event-stream', status=400)


    validation_agent = asave_context["scva_iscca"]

    def generate_full_contract_review_stream():
        try:
            yield stream_event({"event_type": "system_log", "step_code": "FULL_CONTRACT_REVIEW_START", 
                                "message": f"Full contract review initiated for type '{contract_type}'..."})
            time.sleep(0.1)

            # The main work is now a single call to the agent method.
            # This call itself can be long. We submit it to the executor so the Flask request handler
            # for the stream doesn't block, but the client will wait for this future to complete
            # before getting the main result.
            
            # This is where the main LLM call happens and could take time
            yield stream_event({"event_type": "progress", "step_code": "AI_REVIEW_IN_PROGRESS", 
                                "message": "🤖 AI is now performing a holistic review of the entire contract. This may take some time depending on contract length and complexity..."})
            
            review_future = executor.submit(
                process_agent_task, # Your helper
                agent_instance=validation_agent,
                method_name="review_entire_contract",
                task_description=f"FullReview_{contract_type}",
                # Kwargs for review_entire_contract
                contract_text=full_contract_text,
                contract_type=contract_type,
                shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"],
                mined_shariah_rules_path=asave_context["mined_shariah_rules_path"] if os.path.exists(asave_context["mined_shariah_rules_path"]) else None,
                ss_vector_store=asave_context.get("ss_vector_store") # Pass SS store if available
            )

            review_result_wrapper = review_future.result() # This line will block until the review is done

            if review_result_wrapper["status"] == "success":
                full_review_report = review_result_wrapper["result"]
                yield stream_event({"event_type": "full_contract_review_completed", 
                                    "payload": full_review_report})
            else:
                yield stream_event({"event_type": "error", "step_code": "AI_REVIEW_FAILED",
                                    "message": f"AI review of the full contract failed: {review_result_wrapper['error']}",
                                    "payload": {"error_details": review_result_wrapper['error']}})
            
            yield stream_event({"event_type": "full_contract_review_completed", "step_code": "FULL_CONTRACT_STREAM_END", 
                                "message": "Full contract review stream finished."})

        except Exception as e_stream:
            logger.error(f"Error in full contract review stream: {e_stream}", exc_info=True)
            yield stream_event({"event_type": "fatal_error", "step_code": "STREAM_GENERATOR_ERROR", 
                                "message": f"Stream generation failed: {str(e_stream)}"})

    return Response(stream_with_context(generate_full_contract_review_stream()), mimetype='text/event-stream')

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. API will not function correctly.")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True) # threaded=True is important for ThreadPoolExecutor with Flask dev server