from threading import Lock
from flask import Flask, request, jsonify
from time import perf_counter
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from collections.abc import Callable
from typing import Any
from beartype import beartype


@beartype
@dataclass(frozen=True)
class Timestamp:
    start: float
    end: float


@beartype
class JsonRESTServer(ABC):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._call_timestamps = []
        self._call_timestamps_lock = Lock()

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
            print(f"JsonRESTServer.process: calling self._get_response_or_error {data=}")
            result, status_code = self._get_response_or_error(data)

            return jsonify(result), status_code

        @app.route("/get_call_timestamps", methods=["GET"])
        def get_call_timestamps():
            with self._call_timestamps_lock:
                response = [asdict(timestamp) for timestamp in self._call_timestamps]
            return response, 200

        app.run(host=self.host, debug=True, port=self.port)

    def _get_response_or_error(self, arguments: Any) -> tuple[Any, int]:
        start_time = perf_counter()
        try:
            result = self.get_response(**arguments)
        except Exception as e:
            result = (
                {"error": f"Uncaught exception:\n\n{e}\n{traceback.format_exc()}"},
                400,
            )
        end_time = perf_counter()

        with self._call_timestamps_lock:
            self._call_timestamps.append(Timestamp(start=start_time, end=end_time))

        try:
            return result, 200
        except Exception:
            return {"error": f"Unable to convert response to json. {str(result)=}"}, 400

    def get_response(self, **kwargs) -> Any:
        if "function" not in kwargs.keys():
            return {
                "error": 'One of the arguments given to HeadServer.get_response should be called "function"'
            }

        function = kwargs["function"]
        function_kwargs = {key: value for key, value in kwargs.items() if key != "function"}

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
