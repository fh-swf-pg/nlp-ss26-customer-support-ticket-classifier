import re
import unicodedata

WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean and normalize customer support ticket text.

    Steps:
    - normalize Unicode
    - replace line breaks and tabs
    - collapse multiple whitespaces
    - trim leading/trailing whitespace

    Args:
        text: Raw ticket text.

    Returns:
        Cleaned text.
    """

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = (
        text.replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()