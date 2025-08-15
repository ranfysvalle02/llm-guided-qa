#!/usr/bin/env python3
"""
Flask app with a full three-step wizard and multi-turn chat capabilities.
 - Step 1: User provides all inputs.
 - Step 2: AI generates an optimized, editable guide from source files.
 - Step 3: User generates and can edit the complete, final prompt before analysis.
 - Results Page: Includes a Chat tab for follow-up conversation.
 - Evaluations: Now robustly handle errors and always report a status.
"""

import os
import tempfile
import uuid
import logging
import markdown
import json
from datetime import datetime

import ray
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from markupsafe import Markup
from docling.document_converter import DocumentConverter
from openai import AzureOpenAI
from pymongo import MongoClient
from bson import ObjectId

# --- Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-for-flask")
ray.init(ignore_reinit_error=True, runtime_env={"env_vars": dict(os.environ)})

# --- Azure OpenAI Client Configuration ---
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

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

# --- MongoDB Configuration ---
mongo_uri = os.getenv("MONGODB_CONNECTION_URI")
mongo_client = None
db = None
requests_collection = None
evaluations_collection = None
conversations_collection = None

if mongo_uri:
    try:
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client.rag_eval_db
        requests_collection = db.requests
        evaluations_collection = db.evaluations
        conversations_collection = db.conversations
        mongo_client.admin.command('ping')
        logging.info("Successfully connected to MongoDB.")
    except Exception as e:
        logging.error(f"Could not connect to MongoDB. Logging will be disabled. Error: {e}")
        mongo_client = None
else:
    logging.warning("MONGODB_CONNECTION_URI not found. DB logging is disabled.")

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
    "- If the sources do not contain the information to answer the question, you MUST reply with the exact sentence: 'The provided materials do not contain the information to answer this question.'\n"
    "- Answer the user's `QUESTION` directly and concisely. Avoid speculation."
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

def parse_file_to_text(file_storage):
    """
    Parses an uploaded FileStorage object to text.
    Handles .txt and .md files directly and uses docling for other formats.
    This function is corrected using the syntax from the working demo.
    """
    filename = file_storage.filename.lower()
    # Handle simple text files directly
    if filename.endswith((".txt", ".md")):
        try:
            return file_storage.read().decode("utf-8", errors="ignore")
        except Exception as e:
            logging.error(f"Error reading text file {filename}: {e}")
            return ""
    else:
        # For other file types, use the robust docling conversion from the demo
        temp_file_path = None
        try:
            # Create a temporary file to save the upload, ensuring it has the correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                file_storage.save(temp_file.name)
                temp_file_path = temp_file.name

            # Process the file using docling
            converter = DocumentConverter()
            result = converter.convert(temp_file_path)
            # Use the correct export method from the working demo
            markdown_content = result.document.export_to_markdown()
            return markdown_content
        except Exception as e:
            logging.error(f"Error during docling parse of {filename}: {e}")
            return ""
        finally:
            # Ensure the temporary file is cleaned up
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

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
                result["answer"] = output_blocks[content_section_index]["content"][0].get("text", result["answer"])

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

def generate_optimized_guide(raw_guide_text):
    if not global_az_client or not raw_guide_text.strip():
        return raw_guide_text

    guide_system_prompt = """You are an expert technical writer and information architect. Your task is to take a piece of raw, unstructured text (`RAW GUIDE`) and transform it into a well-structured, coherent, and clear set of instructions or reference material.

Your Goal:
- **Clarify and Structure:** Organize the information logically with clear headings, lists, and paragraphs using Markdown.
- **Preserve Information:** Do NOT remove any substantive information, details, or data from the original text. Your job is to restructure, not summarize or omit.
- **Maintain Neutrality:** Do not add new opinions or information not present in the original text. The voice should be objective and informative.
- **Enhance Readability:** Correct grammatical errors, improve sentence structure, and ensure the final text is easy for a human to read and follow.

The output should be a complete, well-formatted markdown document that represents the best possible version of the `RAW GUIDE` content."""

    guide_user_prompt = f"""## RAW GUIDE
{raw_guide_text}

---

Generate the `OPTIMIZED GUIDE` in clear markdown."""

    full_prompt = f"{guide_system_prompt}\n\n{guide_user_prompt}"
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_META", "o4-mini")
    result = get_reasoned_llm_response(global_az_client, full_prompt, model, effort="low")

    return raw_guide_text if "[Error:" in result["answer"] else result["answer"]

def format_history_as_string(history):
    if not history: return ""
    return "\n".join([f"{msg.get('role').capitalize()}: {msg.get('content', '').strip()}" for msg in history])

@ray.remote
def run_evaluation_parallel(
    request_id, optimized_guide_text, doc_text, user_question, expert_answer,
    azure_key, azure_endpoint, azure_api_version, azure_deployment, mongo_db_uri
):
    results = {"request_id": request_id, "timestamp": datetime.utcnow(), "factuality_evaluation_reason": "Evaluation task failed."}
    try:
        if not all([azure_key, azure_endpoint, azure_api_version, azure_deployment]):
            raise ValueError("Missing Azure OpenAI credentials in Ray worker.")

        local_client = AzureOpenAI(api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=azure_key)

        user_prompt = f"## GUIDE\n{optimized_guide_text}\n\n## DOCUMENT\n{doc_text}\n\n---\n\n## QUESTION\n{user_question}"
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        reasoned_result = get_reasoned_llm_response(local_client, full_prompt, azure_deployment)
        generated_output = reasoned_result["answer"]

        results.update({"submitted_answer": generated_output, "expert_answer": expert_answer})

        if "[Error:" not in generated_output:
            eval_prompt = FACTUALITY_JSON_PROMPT_TEMPLATE.format(input=f"Question: {user_question}", expected=expert_answer, output=generated_output)
            eval_result = get_reasoned_llm_response(local_client, eval_prompt, azure_deployment, effort="low")
            json_string = eval_result["answer"].strip().replace("```json", "").replace("```", "")
            jr = json.loads(json_string)
            choice = jr.get("choice", "").upper()
            results.update({
                "factuality_evaluation_choice": choice,
                "factuality_evaluation_reason": jr.get("reason", "No explanation."),
                "factuality_evaluation_score": CHOICE_SCORES.get(choice, 0.0)
            })
        else:
             results["factuality_evaluation_reason"] = "Could not run evaluation because the comparison answer generation failed."

    except Exception as e:
        logging.error(f"FATAL ERROR in run_evaluation_parallel: {e}")
        results["factuality_evaluation_reason"] = f"A critical error occurred in the evaluation task: {e}"

    finally:
        if mongo_db_uri:
            try:
                worker_mongo_client = MongoClient(mongo_db_uri)
                worker_db = worker_mongo_client.rag_eval_db
                worker_db.evaluations.insert_one(results)
                worker_mongo_client.close()
            except Exception as db_e:
                logging.error(f"Ray worker failed to log evaluation to MongoDB: {db_e}")
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
            user_question = request.form.get("user_question", "").strip()
            optimized_guide_content = request.form.get("optimized_guide_content", "").strip()
            final_prompt_text = request.form.get("final_prompt_text", "").strip()
            doc_files = request.files.getlist("document_files")
            reasoning_effort = request.form.get("reasoning_effort", "high")

            original_guide_files = request.files.getlist("guide_files")
            original_guide_text = "".join([parse_file_to_text(f) for f in original_guide_files])
            doc_text = "".join([parse_file_to_text(f) for f in doc_files])

            if not user_question or (not doc_text and "DOCUMENT" in final_prompt_text):
                error = "A question and document are required."
            else:
                try:
                    request_id = None
                    final_prompt_to_use = final_prompt_text
                    if not final_prompt_to_use:
                        user_prompt = f"## GUIDE\n{optimized_guide_content}\n\n## DOCUMENT\n{doc_text}\n\n---\n\n## QUESTION\n{user_question}"
                        final_prompt_to_use = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

                    reasoned_result = get_reasoned_llm_response(global_az_client, final_prompt_to_use, deployment, effort=reasoning_effort)
                    raw_result = reasoned_result["answer"]

                    if requests_collection is not None and conversations_collection is not None:
                        req_log_entry = {
                            "timestamp": datetime.utcnow(), "original_guide_text": original_guide_text,
                            "optimized_guide_submitted": optimized_guide_content, "final_prompt_submitted": final_prompt_to_use,
                            "doc_text": doc_text, "user_question": user_question, "final_answer": raw_result,
                            "reasoning_summaries": reasoned_result["summaries"], "model_used": reasoned_result.get("model") or deployment
                        }
                        insert_result = requests_collection.insert_one(req_log_entry)
                        request_id = str(insert_result.inserted_id)

                        conversations_collection.insert_one({"request_id": request_id, "created_at": datetime.utcnow(), "history": []})

                    if request_id:
                        run_evaluation_parallel.remote(
                            request_id, optimized_guide_content, doc_text, user_question, raw_result,
                            subscription_key, endpoint, api_version, deployment, mongo_uri
                        )

                    result_data = {"guide_text": optimized_guide_content, "doc_text": doc_text, "user_question": user_question, "raw_result": raw_result, "result": Markup(markdown.markdown(raw_result)), "reasoning_summaries": reasoned_result["summaries"], "request_id": request_id}
                except Exception as e:
                    error = f"Error during analysis: {e}"
    return render_template("index.html", error=error, **result_data)

@app.route("/optimize_guide", methods=["POST"])
def optimize_guide_endpoint():
    if not global_az_client: return jsonify({"error": "Azure client not configured."}), 500
    guide_files = request.files.getlist("guide_files")
    if not guide_files: return jsonify({"optimized_guide": ""})

    raw_guide_text = "".join([parse_file_to_text(f) for f in guide_files])
    optimized_guide = generate_optimized_guide(raw_guide_text)

    return jsonify({"optimized_guide": optimized_guide})

@app.route("/preview_final_prompt", methods=["POST"])
def preview_final_prompt():
    user_question = request.form.get("user_question", "").strip()
    doc_files = request.files.getlist("document_files")
    optimized_guide_text = request.form.get("optimized_guide_content", "").strip()
    if not doc_files or not user_question: return jsonify({"error": "Question and document files are required."}), 400
    doc_text = "".join([parse_file_to_text(f) for f in doc_files])
    user_prompt = f"## GUIDE\n{optimized_guide_text}\n\n## DOCUMENT\n{doc_text}\n\n---\n\n## QUESTION\n{user_question}"
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    return jsonify({"prompt": full_prompt})

@app.route("/chat", methods=["POST"])
def chat():
    if not global_az_client or conversations_collection is None or requests_collection is None:
        return jsonify({"error": "Chat backend is not configured."}), 500

    data = request.get_json()
    request_id_str = data.get("request_id")
    user_message = data.get("user_message")
    if not request_id_str or not user_message:
        return jsonify({"error": "Request ID and message are required."}), 400

    try:
        conversation_doc = conversations_collection.find_one({"request_id": request_id_str})
        if not conversation_doc:
            return jsonify({"error": "Conversation context not found."}), 404

        conversation_id = str(conversation_doc.get('_id'))

        original_request = requests_collection.find_one({"_id": ObjectId(request_id_str)})
        if not original_request:
            return jsonify({"error": "Original request data not found."}), 404

        new_history_entry = {"role": "user", "content": user_message}
        conversations_collection.update_one(
            {"request_id": request_id_str},
            {"$push": {"history": new_history_entry}}
        )

        current_history = conversation_doc.get("history", [])
        current_history.append(new_history_entry)
        history_string = format_history_as_string(current_history)

        base_prompt = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            f"## ORIGINAL CONTEXT\n"
            f"### GUIDE\n{original_request.get('optimized_guide_submitted', '')}\n\n"
            f"### DOCUMENT\n{original_request.get('doc_text', '')}\n\n"
            f"### ORIGINAL QUESTION: {original_request.get('user_question', '')}\n\n"
            f"### INITIAL ANSWER:\n{original_request.get('final_answer', '')}\n\n"
            f"---\n\n"
            f"## CONVERSATION HISTORY\n{history_string}\nAssistant:"
        )

        reasoned_result = get_reasoned_llm_response(global_az_client, base_prompt, deployment, effort="medium")
        ai_reply = reasoned_result["answer"]

        conversations_collection.update_one(
            {"request_id": request_id_str},
            {"$push": {"history": {"role": "assistant", "content": ai_reply}}}
        )

        return jsonify({
            "reply": markdown.markdown(ai_reply),
            "conversation_id": conversation_id
        })

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
        else:
            return jsonify({"status": "pending"}), 202
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)