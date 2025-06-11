import waitress
from flask import Flask, request, jsonify
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from beartype import beartype


@beartype
class JsonRESTServer(ABC):
    def __init__(
        self, host: str, port: int, threads: int = 256, connection_limit: int = 1000
    ) -> None:
        self.host = host
        self.port = port
        self.threads = threads
        self.connection_limit = connection_limit

    @abstractmethod
    def functions_exposed_through_api(self) -> dict[str, Callable]:
        pass

    def serve(self) -> None:
        app = Flask(__name__)

        @app.route("/process", methods=["POST"])
        def process():
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.get_json()
            result, status_code = self._get_response_or_error(data)

            return jsonify(result), status_code

        # app.run(host=self.host, port=self.port, threaded=True)
        waitress.serve(
            app,
            host=self.host,
            port=self.port,
            threads=self.threads,
            connection_limit=self.connection_limit,
        )

    def _get_response_or_error(self, arguments: Any) -> tuple[Any, int]:
        try:
            result = self.get_response(**arguments)
            status_code = 200
        except Exception as e:
            result = {"error": f"Uncaught exception:\n\n{e}\n{traceback.format_exc()}"}
            status_code = 400

        try:
            return result, status_code
        except Exception:
            return {"error": f"Unable to convert response to json. {str(result)=}"}, 400

    def get_response(self, **kwargs) -> Any:
        if "function" not in kwargs.keys():
            return {
                "error": 'One of the arguments given to HeadServer.get_response should be called "function"'
            }

        function = kwargs["function"]
        function_kwargs = {
            key: value for key, value in kwargs.items() if key != "function"
        }

        if not isinstance(function, str):
            return {
                "error": 'The "function" argument to HeadServer.get_response should be a string.'
            }

        name_to_function = self.functions_exposed_through_api()

        if function not in name_to_function.keys():
            return {
                "error": f"Invalid function '{function}'. Must be one of {', '.join(name_to_function.keys())}"
            }

        result = name_to_function[function](**function_kwargs)
        return result
