"""The dashboard's API client.

Mocks `requests` throughout -- these tests are about how `client.py` translates HTTP
outcomes (a 200, a 4xx/5xx with a JSON `detail`, a dropped connection) into what the
dashboard renders, not about a real network call.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import requests
from client import AskResult, ask, documents, health


def _response(status_code: int, payload: Any) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
    return response


class TestHealth:
    def test_a_healthy_api_returns_the_parsed_body(self) -> None:
        with patch("client.requests.get", return_value=_response(200, {"status": "ok"})):
            assert health() == {"status": "ok"}

    def test_an_unreachable_api_returns_none_rather_than_raising(self) -> None:
        with patch("client.requests.get", side_effect=requests.ConnectionError("refused")):
            assert health() is None


class TestDocuments:
    def test_returns_the_parsed_body(self) -> None:
        body = {"strategy": "fixed", "documents": []}
        with patch("client.requests.get", return_value=_response(200, body)):
            assert documents() == body

    def test_an_unreachable_api_returns_none(self) -> None:
        with patch("client.requests.get", side_effect=requests.Timeout("slow")):
            assert documents() is None


class TestAsk:
    def test_a_200_is_wrapped_as_ok_with_the_answer(self) -> None:
        body = {"answered": True, "text": "hi"}
        with patch("client.requests.post", return_value=_response(200, body)):
            result = ask("what is fastapi", "hybrid")
        assert result == AskResult(ok=True, data=body)

    def test_a_500_carries_the_status_and_detail_in_the_error(self) -> None:
        with patch(
            "client.requests.post",
            return_value=_response(503, {"detail": "the model could not be reached"}),
        ):
            result = ask("what is fastapi", "hybrid")
        assert result.ok is False
        assert result.error is not None
        assert "503" in result.error
        assert "could not be reached" in result.error

    def test_a_dropped_connection_is_reported_without_raising(self) -> None:
        with patch("client.requests.post", side_effect=requests.ConnectionError("refused")):
            result = ask("what is fastapi", "dense_only")
        assert result.ok is False
        assert result.error is not None
        assert "could not reach the API" in result.error

    def test_sends_the_question_and_mode_as_the_json_body(self) -> None:
        mock_post = Mock(return_value=_response(200, {"answered": True}))
        with patch("client.requests.post", mock_post):
            ask("what is fastapi", "dense_only")
        _, kwargs = mock_post.call_args
        assert kwargs["json"] == {"question": "what is fastapi", "mode": "dense_only"}
