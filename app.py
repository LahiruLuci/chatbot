from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
from vector_db import VectorDB
from data_store import populate_vector_db_from_file
import logging

# Configure logging - this will apply to Flask's default logger as well
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Initialize VectorDB and populate it
vector_db = VectorDB()
if vector_db.collection:
    logging.info("VectorDB collection initialized, proceeding to populate.")
    # Assuming social_phobia_info.txt is in the same directory or accessible path
    populate_vector_db_from_file(vector_db_instance=vector_db, file_path="social_phobia_info.txt")
else:
    logging.warning("VectorDB collection is not initialized. Context retrieval will be skipped and data population is aborted.")

# Define system prompt
system_prompt = """
You are a supportive, emotionally intelligent chatbot designed to provide comfort and genuine help. Your primary goal is to make users feel better after talking with you, not worse. If 'Context from knowledge base:' is provided with the user message, prioritize using this information to make your response more relevant and helpful. However, continue to adhere to all other critical guidelines.

CRITICAL GUIDELINES:
1. PROVIDE ACTUAL HELP: When users express discomfort or negative emotions, offer genuine comfort and practical suggestions rather than just asking more questions.

2. RESPECT "I DON'T KNOW" RESPONSES: Never ask the same question again if a user says "I don't know" or similarly indicates uncertainty. Instead, offer a supportive statement and a different approach.

3. EMOTIONAL COMFORT FIRST: Your main purpose is to help users feel calmer and more at ease. Every response should move toward making them feel better.

4. USER-LED TOPICS: Only discuss social anxiety or mental health when the user brings it up first. Never diagnose or assume their condition.

5. BRIEF AND HELPFUL: Keep responses concise (2-3 sentences) but make them substantive and helpful, not just questions.

6. OFFER PRACTICAL SUGGESTIONS: When users express negative feelings, offer a simple, practical tip they could try, without being pushy.

7. AVOID REPETITION: Track what you've already said and never repeat the same questions or advice.

8. GENUINE CONVERSATION: Be warm and conversational, like a supportive friend who actually helps rather than just asking questions.

Remember: After interacting with you, users should feel calmer, supported, and like they received actual help.
"""

# Create endpoint for chat
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Query ChromaDB for relevant documents
        enriched_user_message = user_message
        if vector_db.collection: # Only query if collection is available
            try:
                retrieved_info = vector_db.query_texts(query_text=user_message, n_results=2)
                if retrieved_info and retrieved_info.get('documents') and retrieved_info['documents'][0] and retrieved_info['documents'][0][0]: # Check if there's actual document content
                    context_from_db = " ".join(retrieved_info['documents'][0])
                    enriched_user_message = f"Context from knowledge base: {context_from_db}\n\nUser message: {user_message}"
                    logging.info(f"Retrieved context for user message: '{user_message[:50]}...' - Context: '{context_from_db[:100]}...'")
                else:
                    logging.info(f"No relevant documents found in VectorDB for user message: '{user_message[:50]}...'")
            except Exception as e:
                logging.error(f"Error querying VectorDB: {e}")
                # Keep user_message as is, or handle error as appropriate
        else:
            logging.warning("Skipping VectorDB query as collection is not initialized.")

        # Send request to OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enriched_user_message}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=150  # Keep responses concise
        )
        
        bot_response = response.choices[0].message.content
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True) # Log full traceback
        return jsonify({"error": str(e)}), 500

# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)