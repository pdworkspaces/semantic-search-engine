"""
Document loading module.

Responsibility:

Read documents from external sources.

Currently supported:

TXT files

Later we can add:

PDF
DOCX
CSV
JSON
Websites
Databases
APIs
"""


def load_text_documents(file_path: str) -> list[str]:
    """
    Load documents from a text file.

    Each non-empty line is treated as one document.

    Parameters
    ----------
    file_path:
        Location of the text file.

    Returns
    -------
    list[str]

        Example:

        [
            "Python is used for AI.",
            "FastAPI is used for APIs."
        ]
    """

    # Open file using UTF-8 encoding.
    with open(file_path, "r", encoding="utf-8") as file:

        # Read every line from the file.
        lines = file.readlines()

    documents = []

    # Process every line.
    for line in lines:

        # Remove spaces and newline characters.
        cleaned_line = line.strip()

        # Ignore empty lines.
        if cleaned_line:

            documents.append(cleaned_line)

    return documents