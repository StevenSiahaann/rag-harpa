import requests
from flask import jsonify

def load_endpoint(url: str, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"error": "Invalid JSON response", "content": response.text}

    except requests.exceptions.RequestException as e:
        return {"error": f"Error during API request: {str(e)}"}