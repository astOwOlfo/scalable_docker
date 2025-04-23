import requests
import traceback
from typing import Any
from beartype import beartype


@beartype
class JsonRESTClient:
    server_url: str

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    @property
    def endpoint(self):
        return f"{self.server_url}/process"

    def call_server(self, **kwargs) -> Any:
        try:
            response = requests.post(
                self.endpoint,
                json=kwargs,
                headers={"Content-Type": "application/json"},
                timeout=600,  # we want this big enough to virtually never happen, but we want this because otherwise the training script can freeze forever
            )
        except Exception as e:
            return {
                "error": f"Error communicating with server: {e} {traceback.format_exc()}"
            }

        try:
            parsed_response = response.json()
        except requests.exceptions.JSONDecodeError:
            parsed_response = None
        
        if response.status_code != 200:
            return {
                "error": f"Error communicating with server.\nStatus code: {response.status_code}.\nResponse json: {parsed_response}"
            }

        return parsed_response
