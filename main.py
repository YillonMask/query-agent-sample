import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

app = Flask(__name__)


class QueryResponse(BaseModel):
    query: str
    answer: str

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/query', methods=['POST'])
def create_query():
    try:
        # Extract the query from the request data
        request_data = request.json
        query = request_data.get('query')
        
        # Log the received query
        logging.info(f"Received query: {query}")

        # Define the system prompt
        system_prompt = (
            "You are an AI assistant that provides information about Kubernetes clusters. "
            "Answer the user's questions based on general Kubernetes knowledge. "
            "Do not include identifiers like pod IDs; use names like 'mongodb' instead of 'mongodb-56c598c8fc'."
        )
        
        # Send the query to the OpenAI API
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        # Extract the assistant's answer
        answer = response.choices[0].message.content.strip()
        
        # Log the generated answer
        logging.info(f"Generated answer: {answer}")
        
        # Create the response model
        response_model = QueryResponse(query=query, answer=answer)
        return jsonify(response_model.dict())
    
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
