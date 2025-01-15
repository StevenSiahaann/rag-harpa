import jwt
import time
from jwt import ExpiredSignatureError, DecodeError, InvalidTokenError

def decode_and_check_exp(token):
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        if "exp" in decoded:
            exp = decoded["exp"]
            if exp < int(time.time()):
                return {"error": "Token has expired"}
        return decoded
    except ExpiredSignatureError:
        return {"error": "Token has expired"}
    except DecodeError:
        return {"error": "Token decoding failed"}
    except InvalidTokenError:
        return {"error": "Invalid token"}


def extract_url_from_curl(curl_command):
    """
    Extract the URL from a curl command string.
    """
    try:
        parts = curl_command.split()
        for i, part in enumerate(parts):
            if part.lower() == '-x' or part.lower() == '--url':
                return parts[i + 1]
            elif part.startswith('http'):
                return part
    except Exception as e:
        print(f"Error parsing curl command: {e}")
    return None