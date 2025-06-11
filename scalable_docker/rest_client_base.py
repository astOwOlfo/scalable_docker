import asyncio
import requests
import aiohttp
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


@beartype
class AsyncJsonRESTClient:
    server_url: str
    max_retries: int
    wait_before_retrying_seconds: int | float

    def __init__(
        self,
        server_url: str,
        max_retries: int = 1,
        wait_before_retrying_seconds: int | float = 4,
    ) -> None:
        self.server_url = server_url
        self.max_retries = max_retries
        self.wait_before_retrying_seconds = wait_before_retrying_seconds

    @property
    def endpoint(self):
        return f"{self.server_url}/process"

    async def call_server(self, **kwargs) -> Any:
        for i_retry in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout()
                ) as session:
                    async with session.post(
                        self.endpoint,
                        json=kwargs,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        try:
                            parsed_response = await response.json()
                        except aiohttp.ContentTypeError:
                            parsed_response = None

                        if response.status != 200:
                            return {
                                "error": f"Error communicating with server.\nStatus code: {response.status}.\nResponse json: {parsed_response}"
                            }

                        return parsed_response
            except aiohttp.ClientConnectorError as e:
                print(
                    f"Call to server failed on attempt {i_retry + 1}/{self.max_retries}"
                )
                if i_retry < self.max_retries - 1:
                    await asyncio.sleep(self.wait_before_retrying_seconds)
                    continue
                return {
                    "error": f"Error communicating with server: {e} {traceback.format_exc()}"
                }
            except Exception as e:
                return {
                    "error": f"Error communicating with server: {e} {traceback.format_exc()}"
                }
