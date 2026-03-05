"""
Custom exception classes for the YouTube SEO Insights Generator.
Provides graceful error management for scraping failures, API issues, and validation errors.
"""

import sys
from src.logger import get_logger

logger = get_logger(__name__)


class SEOAppException(Exception):
    """Base exception class for the YouTube SEO application."""

    def __init__(self, error_message: str, error_detail: sys = None):
        super().__init__(error_message)
        self.error_message = self._build_error_message(error_message, error_detail)
        logger.error(self.error_message)

    @staticmethod
    def _build_error_message(error_message: str, error_detail) -> str:
        if error_detail is not None:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb is not None:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                return (
                    f"Error in [{file_name}] at line [{line_number}]: {error_message}"
                )
        return f"Error: {error_message}"

    def __str__(self):
        return self.error_message


class ScrapingException(SEOAppException):
    """Raised when YouTube data scraping fails (e.g., 403 Forbidden, DOM changes)."""
    pass


class APIException(SEOAppException):
    """Raised when the Gemini 2.0 Flash API call fails (rate limits, timeouts, bad responses)."""
    pass


class ValidationException(SEOAppException):
    """Raised when input validation or output parsing fails (e.g., bad JSON from LLM)."""
    pass
