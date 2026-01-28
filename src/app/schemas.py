from typing import TypedDict, Any, List, Dict


class BillingState(Dict):
    clinical_text: str
    retrieval_type: str  
    retrieved_docs: List[Dict[str, Any]]


class RetrievalIntent(TypedDict, total=False):
    procedure_names: List[str]
    procedure_categories: List[str]
    enm_level: str
    possible_modifiers: List[str]
    diagnosis_context: List[str]


class RetrievalPlanItem(TypedDict):
    type: str
    query: str


class BillingAgentState(TypedDict, total=False):
    note_id: int

    # Patient data
    cleaned_patient_data: Dict[str, Any]
    context: str

    # Retrieval intent and plan
    retrieval_intent: RetrievalIntent
    retrieval_plan: List[RetrievalPlanItem]
    retrieval_type: str
    retrieved_docs: List[Dict[str, Any]]

    # LLM
    prompt: str
    llm_response: Dict[str, Any]
    llm_output: Dict[str, Any]

    # Billing logic output
    bill_items: List[Dict[str, Any]]
    enm: Dict[str, Any]

    # Final output
    final_bill: List[Dict[str, Any]]
    final_response: str
