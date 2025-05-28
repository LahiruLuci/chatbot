from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Define system prompt
system_prompt = """
You are a supportive, emotionally intelligent chatbot designed to provide comfort and genuine help. Your primary goal is to make users feel better after talking with you, not worse.

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
            
        # Send request to OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
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
        return jsonify({"error": str(e)}), 500

# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)