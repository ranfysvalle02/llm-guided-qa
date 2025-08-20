#!/usr/bin/env python3
"""
Flask app with a full three-step wizard and multi-turn chat capabilities.
- Automagically handles creation of MongoDB database, collections, and Atlas Search indexes on first run.
- Step 1: User provides inputs and can find/select similar, existing guides.
- Step 2: AI generates an optimized, editable guide from source files.
- Step 3: User can find similar past analyses before generating the final prompt.
- Results Page: Includes a Chat tab and can now find similar reasoning processes.
- Evaluations: Now robustly handle errors and always report a status.
- Hybrid Search: Includes a /hybrid_search endpoint for MongoDB Atlas.
- Context Augmentation: Users can find past analyses and use their content to enrich the prompt for new requests.
"""

import os
import tempfile
import uuid
import logging
import markdown
import json
import time
from datetime import datetime, timezone

import ray
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from markupsafe import Markup
from docling.document_converter import DocumentConverter
from openai import AzureOpenAI
from pymongo import MongoClient
from bson import ObjectId
from pymongo.errors import OperationFailure

# --- Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
app = Flask(__name__)

# --- Environment Variable Loading & Logging ---
logging.info("--- Loading Configuration from Environment Variables ---")

# Flask Secret Key
flask_secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-for-flask")
app.secret_key = flask_secret_key
if flask_secret_key == "dev-secret-key-for-flask":
    logging.warning("FLASK_SECRET_KEY: Not set, using default development key.")
else:
    logging.info("FLASK_SECRET_KEY: Loaded from environment.")

# Azure OpenAI Credentials
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
# Force the embedding deployment to use text-embedding-3-small
embedding_deployment = "text-embedding-3-small"

logging.info(f"AZURE_OPENAI_ENDPOINT: '{endpoint}'")
logging.info(f"AZURE_OPENAI_DEPLOYMENT: '{deployment}'")
logging.info(f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT: '{embedding_deployment}' (Hardcoded)")
logging.info(f"AZURE_OPENAI_API_VERSION: '{api_version}'")

if not subscription_key:
    logging.warning("AZURE_OPENAI_API_KEY: Not found in environment.")
else:
    logging.info("AZURE_OPENAI_API_KEY: Loaded from environment.")

# MongoDB Connection URI
mongo_uri = os.getenv("MONGODB_CONNECTION_URI")
if not mongo_uri:
    logging.warning("MONGODB_CONNECTION_URI: Not found. DB features (logging, chat, search) are disabled.")
else:
    logging.info("MONGODB_CONNECTION_URI: Loaded from environment.")

logging.info("------------------------------------------------------")


ray.init(ignore_reinit_error=True, runtime_env={"env_vars": dict(os.environ)})

# --- Azure OpenAI Client Configuration ---
global_az_client = None
if not subscription_key:
    logging.warning("AZURE_OPENAI_API_KEY not found; global AzureOpenAI client unavailable.")
else:
    try:
        global_az_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        logging.info("Global AzureOpenAI client configured successfully.")
    except Exception as exc:
        logging.error(f"Error initializing global AzureOpenAI: {exc}")
        global_az_client = None

# --- MongoDB Configuration & Automated Setup ---
mongo_client = None
db = None
requests_collection = None
evaluations_collection = None
conversations_collection = None

def setup_mongodb_indexes(db_instance):
    """Checks for and creates/updates required Atlas Search indexes. Assumes collections exist."""
    requests_coll = db_instance.requests
    
    # --- Index Definitions ---
    vector_index_name = "vector_index"
    vector_index_model = {
      "name": vector_index_name,
      "definition": {
        "mappings": {
          "dynamic": False,
          "fields": {
            "_id": {
                "type": "objectId"
            },
            "doc_text_embedding": {"type": "knnVector", "dimensions": 1536, "similarity": "cosine"},
            "original_guide_embedding": {"type": "knnVector", "dimensions": 1536, "similarity": "cosine"},
            "reasoning_embedding": {"type": "knnVector", "dimensions": 1536, "similarity": "cosine"},
            "input_summary_embedding": {"type": "knnVector", "dimensions": 1536, "similarity": "cosine"}
          }
        }
      }
    }

    text_index_name = "text_index"
    text_index_model = {
        "name": text_index_name, 
        "definition": {"mappings": {"dynamic": True}}
    }
    
    required_indexes = {vector_index_name: vector_index_model, text_index_name: text_index_model}

    try:
        existing_indexes_cursor = requests_coll.list_search_indexes()
        existing_indexes = {idx['name']: idx for idx in existing_indexes_cursor}

        for index_name, model in required_indexes.items():
            needs_update = False
            is_new = False

            if index_name not in existing_indexes:
                is_new = True
                logging.warning(f"Atlas Search index '{index_name}' not found. Creating it now...")
            elif existing_indexes[index_name].get('definition') != model['definition']:
                needs_update = True
                logging.warning(f"Atlas Search index '{index_name}' is outdated. Updating it now...")
            else:
                logging.info(f"Atlas Search index '{index_name}' is up to date.")
                continue

            if is_new:
                requests_coll.create_search_index(model=model)
            elif needs_update:
                requests_coll.update_search_index(name=index_name, definition=model['definition'])

            logging.warning("This setup/update may take a few minutes. The server will wait.")
            timeout = time.time() + 300
            while time.time() < timeout:
                index_status_list = list(requests_coll.list_search_indexes(name=index_name))
                if index_status_list and (index_status_list[0].get('status') == 'READY' or index_status_list[0].get('queryable') is True):
                    logging.info(f"✅ Index '{index_name}' is now ready.")
                    break
                logging.info(f"Waiting for index '{index_name}' to build...")
                time.sleep(15)
            else:
                logging.error(f"Timeout waiting for index '{index_name}' to become ready.")

    except OperationFailure as e:
        if "command listSearchIndexes is not supported" in str(e):
             logging.warning("listSearchIndexes command not supported. Assuming a non-Atlas environment. Search features disabled.")
        else:
            logging.error(f"MongoDB operation failed, possibly due to permissions. Search features may be unavailable. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during index setup: {e}")

def initialize_database(client):
    """Gracefully initializes the database and collections if they don't exist."""
    db = client.the_looking_glass
    required_collections = ["requests", "evaluations", "conversations"]
    existing_collections = db.list_collection_names()

    for coll_name in required_collections:
        if coll_name not in existing_collections:
            logging.warning(f"Collection '{coll_name}' not found. Initializing it now...")
            db.create_collection(coll_name)
            logging.info(f"✅ Collection '{coll_name}' created.")
    
    # Now that we know the 'requests' collection exists, we can safely set up indexes.
    setup_mongodb_indexes(db)
    return db

if mongo_uri:
    try:
        mongo_client = MongoClient(mongo_uri)
        mongo_client.admin.command('ping')
        logging.info("✅ Successfully connected to MongoDB.")
        
        # Centralize database and collection setup
        db = initialize_database(mongo_client)
        requests_collection = db.requests
        evaluations_collection = db.evaluations
        conversations_collection = db.conversations
    except Exception as e:
        logging.error(f"Could not connect to or initialize MongoDB. DB features will be disabled. Error: {e}")
        mongo_client = None

# --- Prompt Templates ---
SYSTEM_PROMPT = (
    "You are 'The Crystallizer,' an AI research assistant from 'The Looking Glass'.\n"
    "Your purpose is to provide clear, transparent, and precise answers based *only* on the provided source materials.\n"
    "Your response should be in well-structured markdown."
)

CHAT_SYSTEM_PROMPT = (
    "You are 'The Crystallizer,' continuing a research conversation.\n"
    "Your purpose is to provide clear, transparent, and precise answers based *only* on the provided source materials from the initial context.\n"
    "- Base your answer exclusively on the information within the `ORIGINAL DOCUMENT` and `ORIGINAL GUIDE` sections. Do not infer or use outside knowledge.\n"
    "- If the sources do not contain the information to answer the latest user message, you MUST reply with the exact sentence: 'The provided materials do not contain the information to answer this question.'\n"
    "- Address the final 'User' message in the `CONVERSATION` block directly and concisely. Avoid speculation."
)

FACTUALITY_JSON_PROMPT_TEMPLATE = """You are a JSON-emitting fact-checking bot. Your SOLE function is to compare a `SUBMITTED ANSWER` to a `REFERENCE ANSWER` and return a single, valid JSON object. DO NOT output any text, explanation, or markdown formatting before or after the JSON object.

## Context
- **Question:** {input}
- **Reference Answer:** {expected}
- **Submitted Answer:** {output}

## Task
Evaluate the `SUBMITTED ANSWER` against the `REFERENCE ANSWER`. Your entire response MUST be a raw JSON object with two keys: "choice" and "reason".

### Choices
- **"A"**: The `SUBMITTED ANSWER` is a factually correct subset of the `REFERENCE ANSWER`.
- **"B"**: The `SUBMITTED ANSWER` is a factually correct superset of the `REFERENCE ANSWER`.
- **"C"**: The `SUBMITTED ANSWER` and `REFERENCE ANSWER` are factually identical.
- **"D"**: The `SUBMITTED ANSWER` factually contradicts the `REFERENCE ANSWER`.
- **"E"**: The answers differ superficially but are factually consistent.

### REQUIRED JSON OUTPUT FORMAT
{{
  "choice": "YOUR_CHOICE_HERE",
  "reason": "A one-sentence justification."
}}
"""

CHOICE_SCORES = {"A": 0.4, "B": 0.6, "C": 1.0, "D": 0.0, "E": 1.0}


# --- Helper Functions ---
def get_embedding(client, text, model):
    if not client: return None
    try:
        text = text.replace("\n", " ")
        if not text.strip(): return None 
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        logging.error(f"Error in get_embedding: {e}")
        return None

def parse_file_to_text(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith((".txt", ".md")):
        return file_storage.read().decode("utf-8", errors="ignore")
    else:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                file_storage.save(temp_file.name)
                temp_file_path = temp_file.name
            converter = DocumentConverter()
            result = converter.convert(temp_file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logging.error(f"Error during docling parse of {filename}: {e}")
            return ""
        finally:
            if temp_file_path and os.path.exists(temp_file_path): os.unlink(temp_file_path)

def get_reasoned_llm_response(client, prompt_text, model_deployment, effort="high"):
    if not client: return {"answer": "[Error: OpenAI client not configured]", "summaries": []}

    try:
        response = client.responses.create(input=prompt_text, model=model_deployment, reasoning={"effort": effort, "summary": "auto"})
        response_data = response.model_dump()

        result = {"answer": "Could not extract a final answer.", "summaries": []}
        output_blocks = response_data.get("output", [])

        if output_blocks:
            summary_section = output_blocks[0].get("summary", [])
            if summary_section:
                result["summaries"] = [s.get("text") for s in summary_section if s.get("text")]

            content_section_index = 1 if summary_section else 0
            if len(output_blocks) > content_section_index and output_blocks[content_section_index].get("content"):
                content_list = output_blocks[content_section_index].get("content", [])
                if content_list:
                     result["answer"] = content_list[0].get("text", result["answer"])

            # Fallback to find the first piece of text content if the structure is unexpected
            if result["answer"] == "Could not extract a final answer.":
                for block in output_blocks:
                    if block.get("content"):
                        for content_item in block["content"]:
                            if content_item.get("text"):
                                result["answer"] = content_item["text"]
                                break
                    if result["answer"] != "Could not extract a final answer.":
                        break

        result["answer"] = result["answer"].strip()
        return result

    except Exception as e:
        logging.error(f"Error in get_reasoned_llm_response: {e}")
        return {"answer": f"[Error calling LLM: {e}]", "summaries": []}

def summarize_for_retrieval(client, doc_text, guide_text, model):
    """Generates a non-sensitive summary of inputs for better future retrieval."""
    if not client or not doc_text:
        return "No summary could be generated."
    
    system_prompt = (
        "You are an expert summarization AI. Your task is to create a detailed but completely anonymized summary of the provided input documents for a searchable knowledge base.\n\n"
        "Your goal is to extract and preserve all facts, figures, and essential details while aggressively redacting any information that could identify the specific person, company, or project involved.\n\n"
        "**WHAT TO PRESERVE (The Essence):**\n"
        "- **Quantitative Data:** Retain specific numbers, metrics, and figures that are essential to the document's meaning (e.g., `50ms latency`, `10TB dataset`, `25% increase in efficiency`).\n"
        "- **Key Attributes & Properties:** Keep important configurations, characteristics, model numbers, or named technical/business concepts (e.g., `'a 3-node replica set'`, `'a company in the European retail sector'`).\n"
        "- **Core Goals & Intent:** Clearly state the primary objective or problem being described (e.g., `'The user needs to reduce database read latency'`, `'The document outlines a plan to enter a new market'`).\n\n"
        "**WHAT TO OMIT (The Identity):**\n"
        "- **All Specific Names:** Redact names of people, companies, and projects.\n"
        "- **Contact & Location Info:** Remove addresses, phone numbers, emails, etc.\n"
        "- **System Identifiers:** Omit IP addresses, hostnames, server names, and domains.\n"
        "- **Credentials:** Remove all passwords, API keys, etc.\n"
        "- **Generalize when needed:** Instead of a specific person's name, use their role (e.g., 'the project manager'). Instead of a specific company name, use their industry and region (e.g., 'a German automotive company').\n\n"
        "--- \n"
        "**Example Transformation:**\n\n"
        "**INPUT:**\n"
        "'Project Atlas for Acme Corp needs to reduce the p99 latency of their user-auth service, running on server SRV-AUTH-01 (10.0.1.5), from 200ms to 50ms for their 1.5 million users. Contact is Jane Doe (jane@acme.com).'\n\n"
        "**GOOD OUTPUT:**\n"
        "'An analysis for a large corporation focused on reducing p99 latency for a user authentication service from 200ms to 50ms. The service needs to support a user base of 1.5 million.'"
    )
    
    # This function now only focuses on the document text (the questionnaire)
    full_content = f"## INPUT DOCUMENT\n{doc_text}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_content}
            ]
        )
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        logging.error(f"Error during summarization for retrieval: {e}")
        return "Summary generation failed."

def format_history_as_string(history):
    if not history: return ""
    return "\n".join([f"{msg.get('role').capitalize()}: {msg.get('content', '').strip()}" for msg in history])

@ray.remote
def run_evaluation_parallel(
    request_id, optimized_guide_text, doc_text, user_question, expert_answer,
    azure_key, azure_endpoint, azure_api_version, azure_deployment, mongo_db_uri,
    db_name, collection_name
):
    """
    Runs the evaluation in a separate process using Ray.
    Connects to services using provided credentials and writes to the specified DB collection.
    """
    logging.info(f"Ray worker started for evaluation of request_id: {request_id}")
    results = {"request_id": request_id, "timestamp": datetime.now(timezone.utc), "factuality_evaluation_reason": "Evaluation task failed."}
    try:
        if not all([azure_key, azure_endpoint, azure_api_version, azure_deployment]):
            raise ValueError("Missing Azure OpenAI credentials for Ray worker.")
        
        local_client = AzureOpenAI(api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=azure_key)
        
        full_prompt = f"{SYSTEM_PROMPT}\n\n## GUIDE\n{optimized_guide_text}\n\n## DOCUMENT\n{doc_text}\n\n---\n\n## QUESTION\n{user_question}"
        
        logging.info(f"Ray worker ({request_id}): Generating comparison answer...")
        reasoned_result = get_reasoned_llm_response(local_client, full_prompt, azure_deployment)
        generated_output = reasoned_result["answer"]
        results.update({"submitted_answer": generated_output, "expert_answer": expert_answer})
        
        if "[Error:" not in generated_output:
            eval_prompt = FACTUALITY_JSON_PROMPT_TEMPLATE.format(input=f"Question: {user_question}", expected=expert_answer, output=generated_output)
            
            logging.info(f"Ray worker ({request_id}): Performing factuality evaluation...")
            eval_response = local_client.chat.completions.create(
                model=azure_deployment,
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"}
            )
            eval_answer = eval_response.choices[0].message.content
            jr = json.loads(eval_answer.strip())
            choice = jr.get("choice", "").upper()
            results.update({
                "factuality_evaluation_choice": choice,
                "factuality_evaluation_reason": jr.get("reason", "No explanation."),
                "factuality_evaluation_score": CHOICE_SCORES.get(choice, 0.0)
            })
            logging.info(f"Ray worker ({request_id}): Evaluation complete.")
        else:
             results["factuality_evaluation_reason"] = "Eval failed: comparison answer generation failed."
    except Exception as e:
        logging.error(f"Ray worker ({request_id}): A critical error occurred: {e}")
        results["factuality_evaluation_reason"] = f"A critical error occurred in the evaluation worker: {e}"
    finally:
        if mongo_db_uri and db_name and collection_name:
            try:
                with MongoClient(mongo_db_uri) as client:
                    database = client[db_name]
                    collection = database[collection_name]
                    collection.insert_one(results)
                    logging.info(f"✅ Ray worker ({request_id}): Successfully wrote evaluation to {db_name}.{collection_name}")
            except Exception as db_e:
                logging.error(f"Ray worker ({request_id}): FAILED to log to MongoDB: {db_e}")
    return True

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result_data = {}
    if request.method == "POST":
        if not global_az_client:
            error = "Azure OpenAI client is not configured."
        else:
            try:
                user_question = request.form.get("user_question", "").strip()
                doc_text = "".join([parse_file_to_text(f) for f in request.files.getlist("document_files")])
                if not user_question or not doc_text:
                    raise ValueError("A question and document are required.")
                
                final_prompt_text = request.form.get("final_prompt_text", "").strip()
                optimized_guide_content = request.form.get("optimized_guide_content", "").strip()
                reasoning_effort = request.form.get("reasoning_effort", "high")
                augmented_context = request.form.get("augmented_context", "").strip()
                
                prompt_parts = [SYSTEM_PROMPT]
                if augmented_context:
                    prompt_parts.append(f"## AUGMENTED CONTEXT FROM PAST ANALYSIS\n{augmented_context}")
                if optimized_guide_content:
                    prompt_parts.append(f"## GUIDE\n{optimized_guide_content}")
                prompt_parts.append(f"## DOCUMENT\n{doc_text}")
                prompt_parts.append(f"---\n\n## QUESTION\n{user_question}")
                
                default_prompt = "\n\n".join(prompt_parts)
                final_prompt_to_use = final_prompt_text or default_prompt
                
                reasoned_result = get_reasoned_llm_response(global_az_client, final_prompt_to_use, deployment, reasoning_effort)
                
                request_id = None
                if requests_collection is not None:
                    # The 'optimized_guide_content' textarea is the single source of truth from the frontend.
                    # This ensures that what's logged to the DB is exactly what was used in the prompt.
                    original_guide_text_from_files = "".join([parse_file_to_text(f) for f in request.files.getlist("guide_files")])
                    original_guide_text = optimized_guide_content or original_guide_text_from_files
                    
                    input_summary = summarize_for_retrieval(global_az_client, doc_text, original_guide_text, deployment)
                    logging.info(f"Generated input summary for request.")

                    doc_embedding = get_embedding(global_az_client, doc_text, embedding_deployment)
                    guide_embedding = get_embedding(global_az_client, original_guide_text, embedding_deployment)
                    summary_embedding = get_embedding(global_az_client, input_summary, embedding_deployment)
                    
                    reasoning_text = " ".join(reasoned_result.get("summaries", []))
                    reasoning_embedding = get_embedding(global_az_client, reasoning_text, embedding_deployment)
                    if reasoning_embedding:
                        logging.info("✅ Successfully generated reasoning summary embedding.")

                    req_log_entry = {
                        "timestamp": datetime.now(timezone.utc), "user_question": user_question, "final_answer": reasoned_result["answer"],
                        "input_summary": input_summary,
                        "doc_text": doc_text, "original_guide_text": original_guide_text,
                        "optimized_guide_submitted": optimized_guide_content, "final_prompt_submitted": final_prompt_to_use,
                        "augmented_context": augmented_context,
                        "reasoning_summaries": reasoned_result["summaries"], "model_used": deployment,
                        "doc_text_embedding": doc_embedding, "original_guide_embedding": guide_embedding,
                        "reasoning_embedding": reasoning_embedding, "input_summary_embedding": summary_embedding,
                    }
                    for key in ["doc_text_embedding", "original_guide_embedding", "reasoning_embedding", "input_summary_embedding"]:
                        if not req_log_entry[key]: del req_log_entry[key]

                    insert_result = requests_collection.insert_one(req_log_entry)
                    request_id = str(insert_result.inserted_id)
                    conversations_collection.insert_one({"request_id": request_id, "created_at": datetime.now(timezone.utc), "history": []})
                    
                    # FINAL FIX: Use explicit 'is not None' checks for database objects, as they do not support truth value testing.
                    can_evaluate = all([
                        db is not None,
                        evaluations_collection is not None,
                        subscription_key,
                        endpoint,
                        api_version,
                        deployment,
                        mongo_uri
                    ])

                    if can_evaluate:
                        logging.info(f"Dispatching evaluation task for request_id: {request_id}")
                        run_evaluation_parallel.remote(
                            request_id, optimized_guide_content, doc_text, user_question, reasoned_result["answer"], 
                            subscription_key, endpoint, api_version, deployment, mongo_uri,
                            db.name, evaluations_collection.name
                        )
                    else:
                        logging.warning(f"Skipping evaluation for request_id: {request_id} due to missing configuration (DB, Azure credentials, etc.).")

                result_data = {
                    "guide_text": optimized_guide_content, "doc_text": doc_text, "user_question": user_question, 
                    "raw_result": reasoned_result["answer"], "result": Markup(markdown.markdown(reasoned_result["answer"])), 
                    "reasoning_summaries": reasoned_result["summaries"], "request_id": request_id
                }
            except Exception as e:
                logging.error(f"Error during analysis: {e}", exc_info=True)
                error = f"Error during analysis: {e}"
    return render_template("index.html", error=error, **result_data)

@app.route("/preview_final_prompt", methods=["POST"])
def preview_final_prompt():
    user_question = request.form.get("user_question", "").strip()
    doc_files = request.files.getlist("document_files")
    optimized_guide_text = request.form.get("optimized_guide_content", "").strip()
    augmented_context = request.form.get("augmented_context", "").strip()
    
    if not doc_files or not user_question: return jsonify({"error": "Question and document files are required."}), 400
    doc_text = "".join([parse_file_to_text(f) for f in doc_files])
    
    # Robustly determine the final guide text, prioritizing the textarea which is the frontend's source of truth.
    guide_files_content = "".join([parse_file_to_text(f) for f in request.files.getlist("guide_files")])
    final_guide_text = optimized_guide_text or guide_files_content

    prompt_sections = [SYSTEM_PROMPT]
    if augmented_context:
        prompt_sections.append(f"## AUGMENTED CONTEXT FROM PAST ANALYSIS\n{augmented_context}")
    if final_guide_text:
        prompt_sections.append(f"## GUIDE\n{final_guide_text}")
    prompt_sections.append(f"## DOCUMENT\n{doc_text}")
    prompt_sections.append(f"---\n\n## QUESTION\n{user_question}")

    full_prompt = "\n\n".join(prompt_sections)
    return jsonify({"prompt": full_prompt})

@app.route("/chat", methods=["POST"])
def chat():
    if not global_az_client or conversations_collection is None or requests_collection is None:
        return jsonify({"error": "Chat backend is not configured."}), 500
    data = request.get_json()
    request_id_str, user_message = data.get("request_id"), data.get("user_message")
    if not all([request_id_str, user_message]): return jsonify({"error": "Request ID and message are required."}), 400
    try:
        original_request = requests_collection.find_one({"_id": ObjectId(request_id_str)})
        conversation_doc = conversations_collection.find_one({"request_id": request_id_str})
        if not all([original_request, conversation_doc]): return jsonify({"error": "Context not found."}), 404
        
        history = [
            {"role": "user", "content": original_request.get('user_question', '')},
            {"role": "assistant", "content": original_request.get('final_answer', '')}
        ]
        history.extend(conversation_doc.get("history", []))
        new_user_entry = {"role": "user", "content": user_message}
        
        full_chat_prompt = f"{CHAT_SYSTEM_PROMPT}\n\n## ORIGINAL CONTEXT\n### GUIDE\n{original_request.get('optimized_guide_submitted', 'N/A')}\n\n### DOCUMENT\n{original_request.get('doc_text', 'N/A')}\n\n---\n\n## CONVERSATION\n{format_history_as_string(history)}\n\nUser: {user_message}\n\nAssistant:"
        reasoned_result = get_reasoned_llm_response(global_az_client, full_chat_prompt, deployment, "medium")
        new_ai_entry = {"role": "assistant", "content": reasoned_result["answer"]}
        
        conversations_collection.update_one({"request_id": request_id_str}, {"$push": {"history": {"$each": [new_user_entry, new_ai_entry]}}})
        
        return jsonify({"reply": markdown.markdown(reasoned_result["answer"]), "summaries": reasoned_result.get("summaries", []), "conversation_id": str(conversation_doc['_id'])})
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_evaluation_results/<request_id>", methods=["GET"])
def get_evaluation_results(request_id):
    if evaluations_collection is None: return jsonify({"error": "DB not configured"}), 500
    try:
        eval_doc = evaluations_collection.find_one({"request_id": request_id})
        if eval_doc:
            eval_doc["_id"] = str(eval_doc["_id"]); eval_doc["request_id"] = str(eval_doc["request_id"]); eval_doc["timestamp"] = eval_doc["timestamp"].isoformat()
            return jsonify(eval_doc)
        return jsonify({"status": "pending"}), 202
    except Exception as e: return jsonify({"error": str(e)}), 500

# --- Similarity and Search Endpoints ---
@app.route("/find_similar_guides", methods=["POST"])
def find_similar_guides():
    if not mongo_client or not global_az_client: return jsonify({"error": "Search not configured."}), 503
    data = request.get_json()
    embedding = get_embedding(global_az_client, data.get("guide_text", ""), embedding_deployment)
    if not embedding: return jsonify([])

    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index", "path": "original_guide_embedding", "queryVector": embedding,
            "numCandidates": 50, "limit": 4
        }},
        {"$project": {
            "_id": 0, "original_guide_text": 1, "score": {"$meta": "vectorSearchScore"}
        }}
    ]
    results = list(requests_collection.aggregate(pipeline))
    return jsonify(results)

@app.route("/find_similar_requests", methods=["POST"])
def find_similar_requests():
    if not mongo_client or not global_az_client: return jsonify({"error": "Search not configured."}), 503
    data = request.get_json()
    doc_text = data.get("doc_text", "")
    if not doc_text:
        return jsonify([])

    # Generate a summary of the current document text to find conceptually similar past summaries.
    query_summary = summarize_for_retrieval(global_az_client, doc_text, "", deployment)
    embedding = get_embedding(global_az_client, query_summary, embedding_deployment)
    if not embedding: 
        return jsonify([])
    
    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index", "path": "input_summary_embedding", "queryVector": embedding,
            "numCandidates": 50, "limit": 4
        }},
        {"$project": {
            "_id": {"$toString": "$_id"}, "user_question": 1, "final_answer": 1, 
            "input_summary": 1, "timestamp": 1, "score": {"$meta": "vectorSearchScore"}
        }}
    ]
    results = list(requests_collection.aggregate(pipeline))
    for r in results:
        if 'timestamp' in r: r['timestamp'] = r['timestamp'].isoformat()
    return jsonify(results)

@app.route("/find_similar_reasoning", methods=["POST"])
def find_similar_reasoning():
    if not mongo_client or not global_az_client: return jsonify({"error": "Search not configured."}), 503
    data = request.get_json()
    request_id = data.get("request_id")
    if not request_id: return jsonify({"error": "Request ID is required."}), 400
    
    try:
        source_doc = requests_collection.find_one({"_id": ObjectId(request_id)}, {"reasoning_embedding": 1})
        if not source_doc or "reasoning_embedding" not in source_doc:
            return jsonify([])
        
        pipeline = [
            {"$vectorSearch": {
                "index": "vector_index", "path": "reasoning_embedding",
                "queryVector": source_doc["reasoning_embedding"],
                "filter": {"_id": {"$ne": ObjectId(request_id)}},
                "numCandidates": 50, "limit": 4
            }},
            {"$project": {
                "_id": {"$toString": "$_id"}, "user_question": 1, "reasoning_summaries": 1, 
                "final_answer": 1, "input_summary": 1, 
                "score": {"$meta": "vectorSearchScore"}
            }}
        ]
        results = list(requests_collection.aggregate(pipeline))
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in find_similar_reasoning: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500
        
@app.route("/get_request_details/<request_id>", methods=["GET"])
def get_request_details(request_id):
    if requests_collection is None:
        return jsonify({"error": "Database not configured."}), 500
    try:
        projection = {
            "_id": 0, "user_question": 1, "input_summary": 1,
            "final_answer": 1, "reasoning_summaries": 1
        }
        details = requests_collection.find_one(
            {"_id": ObjectId(request_id)}, projection
        )
        if not details:
            return jsonify({"error": "Request not found."}), 404
        return jsonify(details)
    except Exception as e:
        logging.error(f"Error fetching request details for {request_id}: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route("/hybrid_search", methods=["POST"])
def hybrid_search():
    if not mongo_client or not global_az_client:
        return jsonify({"error": "Hybrid search requires MongoDB and Azure OpenAI."}), 503
    data = request.get_json()
    query = data.get("query", "").strip() # Use .strip() to handle whitespace
    weights = data.get("weights", {"vectorPipeline": 0.7, "fullTextPipeline": 0.3})
    
    # If there is no query, return the 5 most recent requests as a default.
    if not query:
        try:
            logging.info("No query provided to hybrid_search, returning recent documents.")
            recent_requests = requests_collection.find(
                {},
                {
                    "_id": 1, "user_question": 1, "final_answer": 1, "timestamp": 1,
                    "reasoning_summaries": 1, "input_summary": 1,
                }
            ).sort("timestamp", -1).limit(5)
            
            results = []
            for r in list(recent_requests):
                r['_id'] = str(r['_id'])
                if 'timestamp' in r and isinstance(r['timestamp'], datetime):
                    r['timestamp'] = r['timestamp'].isoformat()
                # Add a default scoreDetails object to maintain a consistent data shape for the frontend
                r['scoreDetails'] = {'value': 0.0, 'details': 'recent item'}
                results.append(r)
            return jsonify(results)
        except Exception as e:
            logging.error(f"An unexpected error occurred during recent documents fetch: {e}")
            return jsonify({"error": "An internal server error occurred while fetching recent items."}), 500

    # If there IS a query, proceed with the hybrid search as before.
    try:
        query_embedding = get_embedding(global_az_client, query, embedding_deployment)
        if not query_embedding: return jsonify({"error": "Failed to generate query embedding."}), 500
        
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorPipeline": [{
                                "$vectorSearch": {
                                    "index": "vector_index",
                                    "path": "input_summary_embedding",
                                    "queryVector": query_embedding,
                                    "numCandidates": 100,
                                    "limit": 10
                                }
                            }],
                            "fullTextPipeline": [{
                                "$search": {
                                    "index": "text_index",
                                    "text": {
                                        "query": query,
                                        "path": ["user_question", "final_answer", "input_summary"]
                                    }
                                }
                            }, {"$limit": 10}]
                        }
                    },
                    "combination": { "weights": weights },
                    "scoreDetails": True
                }
            },
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "user_question": 1, "final_answer": 1, "timestamp": 1,
                    "reasoning_summaries": 1, "input_summary": 1,
                    "scoreDetails": {"$meta": "scoreDetails"}
                }
            },
            {"$limit": 5}
        ]
        results = list(requests_collection.aggregate(pipeline))
        for r in results:
            if 'timestamp' in r and isinstance(r['timestamp'], datetime):
                r['timestamp'] = r['timestamp'].isoformat()
        return jsonify(results)
    except OperationFailure as e:
        logging.error(f"Error during hybrid search: {e}")
        if "Unrecognized pipeline stage name: '$rankFusion'" in str(e):
            return jsonify({"error": "Hybrid search with $rankFusion requires MongoDB Atlas 8.1 or higher."}), 501
        return jsonify({"error": f"A database error occurred: {e}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred during hybrid search: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
