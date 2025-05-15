# api_server.py
import os
import json
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ASAVE Core Components
from utils.document_processor import DocumentProcessor
from agents.suggestion_agent import SuggestionAgent
from agents.validation_agent import ValidationAgent
from agents.shariah_rule_miner_agent import ShariahRuleMinerAgent
# from agents.extraction_agent import ExtractionAgent # KEEA can be added if needed

# --- Flask App Initialization and Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_api_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'json'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Logging Configuration ---
# Configure root logger if this is the main entry point
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Ensure logs go to console
logger = logging.getLogger(__name__) # Logger for this specific module

# --- ASAVE Global Context ---
asave_context = {
    "doc_processor": None,
    "fas_vector_store": None,
    "ss_vector_store": None,
    "all_fas_vector_store": None,
    "aisga": None,
    "scva_iscca": None,
    "srma": None,
    "shariah_rules_explicit_path": "shariah_rules_explicit.json",
    "mined_shariah_rules_path": "output_srma_api/shariah_rules_mined_combined.json", # Default API output path
    "initialized": False
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/initialize', methods=['POST'])
def initialize_asave():
    global asave_context
    logger.info("Received /initialize request.")

    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Initialize failed: GOOGLE_API_KEY not set.")
        return jsonify({"status": "error", "message": "GOOGLE_API_KEY environment variable not set."}), 500

    try:
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
                logger.info(f"Saved FAS file: {filepath}")

        ss_filepaths = []
        for file in ss_files_uploaded:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                ss_filepaths.append(filepath)
                logger.info(f"Saved SS file: {filepath}")
        
        current_shariah_rules_explicit_path = asave_context["shariah_rules_explicit_path"]
        if shariah_rules_explicit_file_uploaded and allowed_file(shariah_rules_explicit_file_uploaded.filename):
            filename = secure_filename(shariah_rules_explicit_file_uploaded.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shariah_rules_explicit_file_uploaded.save(filepath)
            current_shariah_rules_explicit_path = filepath
            asave_context["shariah_rules_explicit_path"] = filepath
            logger.info(f"Saved explicit Shari'ah rules file: {filepath}")
        elif not os.path.exists(current_shariah_rules_explicit_path):
            logger.warning(f"Explicit Shari'ah rules file not provided and default not found: {current_shariah_rules_explicit_path}. Creating dummy file.")
            dummy_rules = [{"rule_id": "DUMMY-API-001", "description":"Dummy rule for API init", "validation_query_template": "Is {suggestion_text} compliant given this dummy rule?"}]
            os.makedirs(os.path.dirname(current_shariah_rules_explicit_path) or '.', exist_ok=True)
            with open(current_shariah_rules_explicit_path, "w") as f: json.dump(dummy_rules, f)
            logger.info(f"Created dummy explicit Shari'ah rules file at {current_shariah_rules_explicit_path}")


        persist_db_path = data.get('persist_db_path_base', "./db_store_api")
        os.makedirs(persist_db_path, exist_ok=True)

        logger.info("Initializing DocumentProcessor...")
        asave_context["doc_processor"] = DocumentProcessor() # GOOGLE_API_KEY checked inside

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
                asave_context["all_fas_vector_store"] = asave_context["fas_vector_store"]
                logger.info(f"FAS Vector Store created/loaded. Path: {fas_db_path}")
            else:
                logger.warning("No FAS chunks generated, FAS vector store not created.")
        else:
            logger.warning("No FAS files provided. FAS Vector Store not created.")

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
            else:
                logger.warning("No SS chunks generated, SS vector store not created.")
        else:
            logger.warning("No SS files provided. SS Vector Store not created.")

        logger.info("Initializing Agents...")
        asave_context["aisga"] = SuggestionAgent()
        asave_context["scva_iscca"] = ValidationAgent()
        asave_context["srma"] = ShariahRuleMinerAgent()
        asave_context["initialized"] = True
        logger.info("ASAVE system initialized successfully via API.")
        
        return jsonify({
            "status": "success",
            "message": "ASAVE system initialized.",
            "fas_vector_store_status": "Created/Loaded" if asave_context["fas_vector_store"] else "Not Created",
            "ss_vector_store_status": "Created/Loaded" if asave_context["ss_vector_store"] else "Not Created",
            "explicit_shariah_rules_path": asave_context["shariah_rules_explicit_path"]
        }), 200

    except ValueError as ve: # Catch specific errors like API key missing during agent/doc_proc init
        logger.error(f"Initialization ValueError: {ve}", exc_info=True)
        return jsonify({"status": "error", "message": str(ve)}), 500
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"General initialization error: {str(e)}"}), 500


@app.route('/analyze_chunk', methods=['POST'])
def analyze_chunk_api(): # Renamed to avoid conflict if running tests in same interpreter
    global asave_context
    logger.info("Received /analyze_chunk request.")

    if not asave_context["initialized"]:
        logger.warning("Analyze request received but system not initialized.")
        return jsonify({"status": "error", "message": "ASAVE system not initialized. Call /initialize first."}), 400

    if not all([asave_context["aisga"], asave_context["scva_iscca"]]): # SS vector store can be optional for SCVA if only explicit rules are used
        logger.error("Core components (AISGA, SCVA/ISCCA) missing from context.")
        return jsonify({"status": "error", "message": "Core analysis components (AISGA, SCVA/ISCCA) not available."}), 500

    try:
        data = request.get_json()
        if not data:
            logger.warning("Analyze request with invalid JSON payload.")
            return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

        target_text = data.get("target_text_chunk")
        fas_context_strings = data.get("fas_context_chunks", [])
        ss_context_strings = data.get("ss_context_chunks", [])
        fas_name = data.get("fas_name_for_display", "Unnamed FAS")
        ambiguity_desc = data.get("identified_ambiguity", "The provided text section is under review for potential clarification or enhancement.")

        if not target_text:
            logger.warning("Analyze request missing 'target_text_chunk'.")
            return jsonify({"status": "error", "message": "Missing 'target_text_chunk' in request."}), 400

        logger.info(f"AISGA processing for target: '{target_text[:50]}...'")
        aisga_full_output = asave_context["aisga"].generate_clarification(
            original_text=target_text,
            identified_ambiguity=ambiguity_desc,
            fas_context_strings=fas_context_strings,
            ss_context_strings=ss_context_strings
        )
        
        # Check if AISGA returned an error structure
        if isinstance(aisga_full_output, dict) and "error" in aisga_full_output:
            logger.error(f"AISGA returned an error: {aisga_full_output['error']}")
            return jsonify({"status": "error", "message": f"AISGA processing error: {aisga_full_output['error']}", "details": aisga_full_output}), 500
        if not aisga_full_output or not isinstance(aisga_full_output, dict): # Basic check
             logger.error(f"AISGA failed to generate a valid suggestion object. Output: {aisga_full_output}")
             return jsonify({"status": "error", "message": "AISGA failed to generate a valid suggestion object."}), 500


        aisga_prompt_details = aisga_full_output.pop("prompt_details_actual", {})
        aisga_structured_suggestion = aisga_full_output

        logger.info(f"SCVA processing for suggestion: '{aisga_structured_suggestion.get('proposed_text', '')[:50]}...'")
        scva_report = asave_context["scva_iscca"].validate_shariah_compliance(
            proposed_suggestion_object=aisga_structured_suggestion,
            shariah_rules_explicit_path=asave_context["shariah_rules_explicit_path"],
            ss_vector_store=asave_context["ss_vector_store"],
            mined_shariah_rules_path=asave_context["mined_shariah_rules_path"] if os.path.exists(asave_context["mined_shariah_rules_path"]) else None
        )
        logger.info(f"SCVA Report Status: {scva_report.get('overall_status')}")

        logger.info(f"ISCCA processing for suggestion: '{aisga_structured_suggestion.get('proposed_text', '')[:50]}...'")
        iscca_report = asave_context["scva_iscca"].validate_inter_standard_consistency(
            proposed_suggestion_object=aisga_structured_suggestion,
            fas_name=fas_name,
            all_fas_vector_store=asave_context["all_fas_vector_store"]
        )
        logger.info(f"ISCCA Report Status: {iscca_report.get('status')}")

        thinking_process_response = {
            "input_summary": {
                "target_text_chunk_len": len(target_text),
                "num_fas_context_chunks": len(fas_context_strings),
                "num_ss_context_chunks": len(ss_context_strings),
                "fas_name": fas_name,
                "ambiguity_desc_len": len(ambiguity_desc)
            },
            "aisga_step": {
                "prompt_details": aisga_prompt_details,
                "structured_suggestion": aisga_structured_suggestion
            },
            "scva_step": { "structured_report": scva_report },
            "iscca_step": { "structured_report": iscca_report }
        }
        logger.info("Chunk analysis complete. Returning structured response.")
        return jsonify({"status": "success", "analysis": thinking_process_response}), 200

    except Exception as e:
        logger.error(f"Error during chunk analysis: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"General analysis error: {str(e)}"}), 500


@app.route('/mine_shariah_rules', methods=['POST'])
def mine_shariah_rules_api(): # Renamed
    global asave_context
    logger.info("Received /mine_shariah_rules request.")

    if not asave_context["initialized"] or not asave_context["srma"] or not asave_context["doc_processor"]:
        logger.warning("SRMA request received but system/SRMA components not initialized.")
        return jsonify({"status": "error", "message": "ASAVE system or SRMA components not initialized."}), 400

    try:
        data = request.form
        uploaded_files = request.files.getlist('ss_files_for_srma')
        
        ss_metadata_list = []
        for i, file in enumerate(uploaded_files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Saved SS file for SRMA: {filepath}")
                
                full_name = data.get(f'ss_files_for_srma_{i}_fullname', f"Unnamed Standard {os.path.splitext(filename)[0]}")
                short_code = data.get(f'ss_files_for_srma_{i}_shortcode', f"USS{i+1}")
                
                ss_metadata_list.append({
                    "filepath": filepath,
                    "standard_name_full": full_name,
                    "standard_short_code": short_code.upper()
                })

        if not ss_metadata_list:
            logger.warning("No valid SS files or metadata provided for SRMA.")
            return jsonify({"status": "error", "message": "No valid SS files or metadata provided for SRMA."}), 400

        output_dir = data.get('output_directory', "output_srma_api")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"SRMA processing {len(ss_metadata_list)} SS files into {output_dir}...")
        combined_rules_path = asave_context["srma"].mine_rules_from_document_list(
            ss_documents_with_metadata=ss_metadata_list,
            doc_processor_instance=asave_context["doc_processor"],
            base_output_dir=output_dir
        )
        
        if combined_rules_path and os.path.exists(combined_rules_path):
            asave_context["mined_shariah_rules_path"] = combined_rules_path
            logger.info(f"SRMA processing complete. Output: {combined_rules_path}")
            return jsonify({
                "status": "success",
                "message": "SRMA processing complete.",
                "output_file_path": combined_rules_path,
                "num_files_processed": len(ss_metadata_list)
            }), 200
        else:
            logger.error(f"SRMA failed to produce an output file. Expected at: {combined_rules_path}")
            return jsonify({"status": "error", "message": "SRMA failed to produce an output file."}), 500

    except Exception as e:
        logger.error(f"Error during SRMA execution: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"General SRMA error: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status_api(): # Renamed
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
            "aisga": bool(asave_context["aisga"]),
            "scva_iscca": bool(asave_context["scva_iscca"]),
            "srma": bool(asave_context["srma"]),
        }
    })

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. API will not function correctly.")
        # Optionally exit or raise an error here if you want to prevent startup without API key
        # raise EnvironmentError("GOOGLE_API_KEY must be set to run the ASAVE API.")
    
    app.run(host='0.0.0.0', port=5001, debug=False) # Set debug=False for more production-like logging