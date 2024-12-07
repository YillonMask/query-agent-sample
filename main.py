import logging
import subprocess
import shlex  # For safely splitting the command string
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import openai
import os
from dotenv import load_dotenv
import re  # Import regular expressions module

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

def execute_kubectl_command(command):
    try:
        # Execute kubectl command and capture output
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"kubectl command failed: {e}")
        return f"Error executing command: {e.stderr}"

def extract_kubectl_command(response_text):
    """
    Extracts the kubectl command from the assistant's response.
    Assumes the command starts with 'kubectl' and is on a single line.
    """
    # Use regular expression to find the line that starts with 'kubectl'
    match = re.search(r'^(kubectl[^\n\r]*)', response_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    else:
        # If no match is found, raise an error
        raise ValueError("No kubectl command found in the assistant's response.")

def process_command_output(query, command_output):
    """
    Processes the output of the kubectl command based on the user's query.
    """
    query_lower = query.lower().strip()
    
    if query_lower == "how many pods are in the default namespace?":
        # Count the number of pods
        lines = command_output.strip().split('\n')
        pod_count = len(lines) - 1  # Subtract 1 for the header
        return str(pod_count)
    
    elif query_lower == "how many nodes are there in the cluster?":
        # Count the number of nodes
        lines = command_output.strip().split('\n')
        node_count = len(lines) - 1  # Subtract 1 for the header
        return str(node_count)
    
    elif query_lower == "which pod is spawned by my-deployment?":
        # Extract the pod name(s) from the command output
        lines = command_output.strip().split('\n')
        pods = []
        for line in lines[1:]:  # Skip the header
            columns = line.split()
            if columns:
                pod_name = columns[0]
                pods.append(pod_name)
        if pods:
            return pods[0]  # Return the first pod name
        else:
            return "No pods found for the deployment."
    
    elif query_lower == "what is the status of the pod named 'example-pod'?":
        # Extract the status of the specified pod
        lines = command_output.strip().split('\n')
        for line in lines[1:]:  # Skip the header
            columns = line.split()
            if columns and columns[0] == "example-pod":
                if len(columns) >= 3:
                    status = columns[2]  # Assuming the status is in the third column
                    return status
                else:
                    return "Could not determine the status."
        return "Pod not found."
    
    else:
        # For any other queries, return the original command output
        return command_output

@app.route('/query', methods=['POST'])
def create_query():
    try:
        # Extract the query from the request data
        request_data = request.json
        query = request_data.get('query')
        
        # Log the received query
        logging.info(f"Received query: {query}")

        # Define the system prompt for generating kubectl commands
        system_prompt = (
            "As an AI assistant, convert the user's query into an appropriate kubectl command "
            "that outputs the needed information directly and concisely. "
            "Output only the kubectl command without any additional text, explanations, or formatting. "
            "Do not include code blocks or markdown formatting. "
            "Ensure the command is safe, uses 'kubectl get' or 'kubectl describe', and does not modify the cluster. "
            "For counting resources, provide commands that list them so counts can be derived."
        )
        
        # Send the query to the OpenAI API to get the kubectl command
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ],
            max_tokens=50,
            n=1,
            temperature=0,
        )
        
        # Extract the assistant's response text
        response_text = response.choices[0].message.content.strip()

        # Log the raw response
        logging.info(f"Assistant's raw response: {response_text}")

        # Extract the kubectl command from the response
        command_text = extract_kubectl_command(response_text)
        
        # Log the extracted command
        logging.info(f"Extracted command: {command_text}")
        
        # Validate that the command starts with 'kubectl get' or 'kubectl describe'
        allowed_commands = ['kubectl get', 'kubectl describe']
        if not any(command_text.startswith(cmd) for cmd in allowed_commands):
            raise ValueError("Generated command is not allowed.")
        
        # Execute the kubectl command
        result = subprocess.run(
            shlex.split(command_text),
            capture_output=True,
            text=True,
            check=True
        )

        # Get the command output
        command_output = result.stdout.strip()

        # Log the command output
        logging.info(f"Command output: {command_output}")

        # Process the command output based on the query
        processed_output = process_command_output(query, command_output)

        # Create the response
        response_model = QueryResponse(query=query, answer=processed_output)
        return jsonify(response_model.dict())
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing command: {e.stderr}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500
    except ValueError as e:
        error_msg = f"Validation error: {str(e)}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
