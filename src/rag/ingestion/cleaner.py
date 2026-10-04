"""Document cleaning utilities."""

import re
from collections import Counter
from langchain_core.documents import Document


# -------------------------------------------------------------------
# Ebook / PDF boilerplate
# -------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    r"thank you for downloading this",
    r"^simon & schuster ebook\.?$",
    r"ebook when you join our mailing list",
    r"sign up and see terms and conditions",
    r"^and see terms and conditions\.?$",
    r"click here to sign up",
    r"click below to sign up",
    r"already a subscriber",
    r"you will continue to receive exclusive offers",
    r"deals, recommended reads, and more from simon & schuster",
    r"^your inbox\.?$",
    r"oceanofpdf",
    r"downloaded from",
    r"download free ebook",
    r"this ebook is provided",
]


# -------------------------------------------------------------------
# Patterns that identify pages which contain almost no useful content
# -------------------------------------------------------------------

PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:page\s*)?\d+\s*$",
    re.IGNORECASE,
)

# Common ebook page markers such as:
# "1", "Page 1", "- 1 -", "1 of 100"
PAGE_MARKER_PATTERN = re.compile(
    r"^\s*(?:page\s*)?\d+(?:\s+of\s+\d+)?\s*$",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace without destroying paragraph structure.
    """

    # Normalize different newline formats
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse excessive spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def remove_boilerplate(text: str) -> str:
    """
    Remove known ebook/download boilerplate lines from the text.
    """

    cleaned_lines = []

    for line in text.splitlines():
        line_lower = line.strip().lower()

        if any(
            re.search(pattern, line_lower)
            for pattern in BOILERPLATE_PATTERNS
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_page_number_only(text: str) -> bool:
    """
    Detect pages containing only a page number.
    """

    text = text.strip()

    if not text:
        return True

    return bool(PAGE_MARKER_PATTERN.fullmatch(text))


def remove_repeated_headers_footers(
    documents: list[Document],
    min_occurrences: int = 5,
) -> list[Document]:
    """
    Remove lines that occur repeatedly across many pages.

    Useful for PDF headers/footers such as:

        Elon Musk
        Walter Isaacson

    or:

        CHAPTER 3
        42

    We only remove a line when it occurs frequently enough
    to be considered a repeated PDF artifact.
    """

    line_counts = Counter()

    # ---------------------------------------------------------------
    # First pass: count individual lines across documents
    # ---------------------------------------------------------------

    for doc in documents:
        lines = {
            line.strip()
            for line in doc.page_content.splitlines()
            if line.strip()
        }

        for line in lines:
            line_counts[line] += 1

    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= min_occurrences
        and len(line) <= 150
    }

    # ---------------------------------------------------------------
    # Second pass: remove repeated lines
    # ---------------------------------------------------------------

    cleaned_documents = []

    for doc in documents:

        lines = doc.page_content.splitlines()

        cleaned_lines = [
            line
            for line in lines
            if line.strip() not in repeated_lines
        ]

        cleaned_text = "\n".join(cleaned_lines)

        cleaned_documents.append(
            Document(
                page_content=cleaned_text,
                metadata=doc.metadata.copy(),
            )
        )

    return cleaned_documents


def clean_document(doc: Document) -> Document | None:
    """
    Clean one LangChain Document.

    Returns:
        Document -> if useful content remains
        None     -> if the document should be discarded
    """

    text = doc.page_content

    # ---------------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------------

    text = normalize_whitespace(text)

    # ---------------------------------------------------------------
    # Remove completely empty pages
    # ---------------------------------------------------------------

    if not text:
        return None

    # ---------------------------------------------------------------
    # Remove pages that contain only page numbers
    # ---------------------------------------------------------------

    if is_page_number_only(text):
        return None

    # ---------------------------------------------------------------
    # Remove obvious ebook/download boilerplate
    # ---------------------------------------------------------------

    text = remove_boilerplate(text)
    text = normalize_whitespace(text)

    if not text:
        return None

    # ---------------------------------------------------------------
    # Create cleaned document
    # ---------------------------------------------------------------

    metadata = doc.metadata.copy()

    return Document(
        page_content=text,
        metadata=metadata,
    )


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Clean the complete list of LangChain Documents.

    Pipeline:

        Raw Documents
              ↓
        Basic cleaning
              ↓
        Remove empty/noise pages
              ↓
        Remove repeated headers/footers
              ↓
        Clean Documents
    """

    cleaned_documents = []

    # ---------------------------------------------------------------
    # Step 1: Clean individual documents
    # ---------------------------------------------------------------

    for document in documents:

        cleaned_document = clean_document(document)

        if cleaned_document is not None:
            cleaned_documents.append(cleaned_document)

    # ---------------------------------------------------------------
    # Step 2: Remove repeated headers and footers
    # ---------------------------------------------------------------

    cleaned_documents = remove_repeated_headers_footers(
        cleaned_documents,
        min_occurrences=5,
    )

    # ---------------------------------------------------------------
    # Step 3: Remove documents that became empty
    # ---------------------------------------------------------------

    final_documents = []

    for document in cleaned_documents:

        document.page_content = normalize_whitespace(
            document.page_content
        )

        if document.page_content:
            final_documents.append(document)

    return final_documents