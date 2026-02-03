import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.services.patient_data import get_notes
from utils.helper import clean_html_tags
from loguru import logger





async def prepare_patient_data(state: dict) -> dict:
    """
    Prepares and cleans patient data by removing HTML tags
    from specific fields in the notes.
    """
    note_id = state.get("note_id")
    if not note_id:
        raise ValueError("note_id is required in state")


    raw_data =  await get_notes(note_id)
    logger.info(f"Raw patient data retrieved {raw_data}")
    cleaned_data = clean_html_tags(raw_data) #type: ignore
    
    return {
        **state,
        "cleaned_patient_data": cleaned_data
    }



def context_builder(cleaned_data: dict) -> str:
    """
    Builds a context string from cleaned patient data.
    """
    if not cleaned_data or not isinstance(cleaned_data, dict):
        return ""


    note_id = next(iter(cleaned_data.keys()))
    note = cleaned_data[note_id].get("note", {})
    other_notes = cleaned_data[note_id]
    
    biopsy_notes = note.get("biopsyNotes", "")
    examination = note.get("examination", "")
    patient_summary = note.get("patientSummary", "")
    complaints = note.get("complaints", "")
    procedure = note.get("procedure", "")
    current_medication = note.get("currentmedication", "")
    place_of_service = note.get("PlaceOfService", "")
    patient_history = note.get("pastHistory", "")
    mohs_notes = note.get("mohsNotes", "")
    note_date = note.get("noteDate", "")
    assessment = note.get("assesment","")
    diagnoses = note.get("diagnoses", "")
    allergy = note.get("allergy", "")
    biopsy = other_notes.get("biopsy", []) or []
    general = other_notes.get("general", []) or []
    mohs = other_notes.get("mohs", []) or []
    prescription = other_notes.get("prescriptions", []) or []
    previous_superbill = other_notes.get("previous_superbill", []) or []
    meds = [
        f"{rx.get('brandName', '')} {rx.get('strength', '')}".strip()
        for rx in prescription
        if isinstance(rx, dict)
    ]
    
    
    biopsy_section = ""
    if biopsy:
        biopsy_section = "\n[BIOPSY PROCEDURES - STRUCTURED DATA]\n"
        for idx, bx in enumerate(biopsy, 1):
            biopsy_section += f"""
            Biopsy #{idx}:
            - Procedure: {bx.get('proName', 'N/A')}
            - Site: {bx.get('site', 'N/A')}
            - Location: {bx.get('location', 'N/A')}
            - Lesion Size: {bx.get('lesionSize', 'N/A')} mm
            - Method: {bx.get('method', 'N/A')}
            - Wound Size: {bx.get('woundSize', 'N/A')} mm
            - Closure Size: {bx.get('closureSize', 'N/A')} mm
            - Rule Out Dx: {bx.get('ruleOutDx', 'N/A')}
            """
            
    general_section = ""
    if general:
        general_section = "\n[GENERAL PROCEDURES - STRUCTURED DATA]\n"
        for idx, proc in enumerate(general, 1):
            general_section += f"""
            Procedure #{idx}:
            - Name: {proc.get('proName', 'N/A')}
            - Site: {proc.get('site', 'N/A')}
            - Location: {proc.get('location', 'N/A')}
            - Quantity: {proc.get('qty', 'N/A')}
            - Billing Size: {proc.get('billingSize', 'N/A')} mm
            - Method: {proc.get('method', 'N/A')}
            """
    mohs_section = ""
    if mohs:
        mohs_section = "\n[MOHS PROCEDURES - STRUCTURED DATA]\n"
        for idx, m in enumerate(mohs, 1):
            mohs_section += f"""
            MOHS #{idx}:
            - Procedure: {m.get('proName', 'N/A')}
            - Site: {m.get('site', 'N/A')}
            - Location: {m.get('location', 'N/A')}
            - Pre-Op Size: {m.get('preOpSize', 'N/A')} mm
            - Post-MOHS Size: {m.get('postMohsSize', 'N/A')} mm
            - Post-Repair Size: {m.get('postRepairSize', 'N/A')} mm
            - Repair Type: {m.get('repairId', 'N/A')}
            """
            
            
    prev_sb_section = ""
    if previous_superbill:
        prev_sb_section = "\n[PREVIOUS SUPERBILL REFERENCE]\n"
        for item in previous_superbill:
            prev_sb_section += (
                f"- {item.get('Procedure', 'N/A')} "
                f"(CPT: {item.get('CPT', 'N/A')}) "
                f"x {item.get('Quantity', 1)}\n"
                f"Charge per unit = {item.get('Charge Per Unit', 'N/A')}"
            )
            
    context = f"""
    ### PATIENT ENCOUNTER SUMMARY ###
    NOTE_ID: {note_id}
    PATIENT_STATUS: {patient_summary}
    PLACE_OF_SERVICE: {place_of_service}
    PATIENT_HISTORY: {patient_history}
    CURRENT_MEDICATION: {current_medication}
    ALLERGY: {allergy}
    ENCOUNTER_DATE: {note_date}
    [CHIEF COMPLAINT]
    {complaints}
    [PHYSICAL EXAMINATION & SITE FINDINGS]
    {examination}
    [ASSESSMENT & CLINICAL DIAGNOSES]
    {assessment}
    ICD-10 CODES: {diagnoses}
    {biopsy_section}
    {general_section}
    {mohs_section}
    [UNSTRUCTURED PROCEDURE NOTES]
    - Procedure Field: {procedure}
    - Biopsy Notes: {biopsy_notes}
    - MOHS Notes: {mohs_notes}
    [PLAN & MEDICATIONS]
    Prescribed Today: {", ".join(meds) if meds else "None"}
    {prev_sb_section}
    """
    logger.info(f"preparing context for patient {context}")

    return context.strip()


