import requests
def load_endpoint(url: str, headers):
    """
    Mengambil data dari endpoint API dan mengembalikan respons JSON.
    Menangani error jika terjadi masalah saat memuat endpoint.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"error": "Invalid JSON response from the server."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Error during API request: {str(e)}"}