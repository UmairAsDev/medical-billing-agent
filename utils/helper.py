import re
import copy
from bs4 import BeautifulSoup
from loguru import logger   

def html_parser(html_content: str) -> str:
    if not isinstance(html_content, str):
        return ""
    text = BeautifulSoup(html_content, "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()




def clean_html_tags(data: dict) -> dict:
    """
    Returns a cleaned copy of patient data without mutating original.
    """
    try:
        if not data or not isinstance(data, dict):
            raise ValueError("Invalid data passed to clean_html_tags_safe")

        cleaned = copy.deepcopy(data)
        note_id = next(iter(cleaned.keys()))

        fields_to_clean = [
            "biopsyNotes",
            "examination",
            "patientSummary",
            "complaints",
            "currentmedication",
            "pastHistory",
            "reviewofsystem",
            "assesment",
            "procedure",
            "mohsNotes",
            "allergy",
        ]

        note = cleaned[note_id].get("note", {})

        for field in fields_to_clean:
            if field in note and note[field]:
                note[field] = html_parser(note[field])

        return cleaned

    except Exception as e:
        logger.error(f"Error in clean_html_tags: {e}")
        return {}



def constructor():
    pass