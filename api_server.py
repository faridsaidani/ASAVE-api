# api_server.py
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
    "initialized": False
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

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. API will not function correctly.")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True) # threaded=True is important for ThreadPoolExecutor with Flask dev server