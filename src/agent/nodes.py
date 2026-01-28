import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from config.gunicorn_conf import settings
from src.services.prepare_data import context_builder, prepare_patient_data
from src.services.llm_factory import get_openai_llm
from langchain_core.messages import AIMessage
from loguru import logger
load_dotenv()

OpenAI(api_key =settings.OPENAI_API_KEY)


embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    collection_name="billing_knowledge",
    embedding_function=embedding,
    persist_directory="./vectordb/chroma"
)





def prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    procedure_rules = []
    modifier_rules = []
    enm_rules = []
    retrieved_docs = state.get("retrieved_docs", [])

    for d in retrieved_docs:
        meta = d.get("metadata", {})
        content = d.get("content", "")

        doc_type = meta.get("type", "")
        if doc_type == "procedure":
            procedure_rules.append(meta)
        elif doc_type == "modifier":
            # Include both metadata and the full content which has detailed rules
            modifier_rules.append({
                "modifier": meta.get("modifier"),
                "enmModifier": meta.get("enmModifier"),
                "details": content  # Full text with modifierDetDesc
            })
        elif doc_type == "enm":
            enm_rules.append(meta)

    # Get retrieval intent for additional context
    intent = state.get("retrieval_intent", {})
    procedure_names = intent.get("procedure_names", [])
    diagnosis_context = intent.get("diagnosis_context", [])
    possible_modifiers = intent.get("possible_modifiers", [])

    # Format modifier rules as readable text
    modifier_rules_text = ""
    for mod in modifier_rules:
        modifier_rules_text += f"\n**Modifier {mod.get('modifier')}** (E/M applicable: {mod.get('enmModifier', False)}):\n{mod.get('details', '')}\n"

    # Check if vectorstore has incomplete data
    has_incomplete_data = state.get("has_incomplete_vectorstore_data", False)
    
    # Extract previous superbill - ONLY use as fallback when vectorstore is incomplete
    # Note: patient data is stored under 'cleaned_patient_data' key
    cleaned_patient_data = state.get("cleaned_patient_data", {})
    # Get the first note's data (keyed by note_id)
    note_id = state.get("note_id")
    note_data = cleaned_patient_data.get(note_id, {}) if note_id else {}
    previous_superbill = note_data.get("previous_superbill", [])
    previous_code_map = state.get("previous_code_map", {})
    
    # Build previous superbill mapping ONLY if vectorstore has incomplete data
    prev_superbill_text = ""
    if has_incomplete_data and previous_superbill:
        prev_superbill_text = "\n### FALLBACK: PREVIOUS SUPERBILL MAPPING\n"
        prev_superbill_text += "**NOTE**: Vectorstore has incomplete procedure descriptions. Using previous superbill as reference:\n"
        for item in previous_superbill:
            proc_name = item.get("Procedure", "")
            cpt = item.get("CPT", "")
            mod = item.get("modifierId", "")
            qty = item.get("Quantity", 1)
            charge_per_unit = item.get("Charge Per Unit", "NO")
            dx = item.get("GROUP_CONCAT(dc.icd10Code SEPARATOR ', ')", "")
            prev_superbill_text += f"- {proc_name} → CPT: {cpt}"
            if mod:
                prev_superbill_text += f" (Modifier: {mod})"
            prev_superbill_text += f" Qty: {qty}, ChargePerUnit: {charge_per_unit}, Dx: {dx}\n"
        prev_superbill_text += "\n**IMPORTANT**: If the current note has the same procedure names, use the EXACT CPT codes from above.\n"

    # Extract context
    context = state.get("context", "")
    
    # Build the prompt based on whether we have complete vectorstore data
    if has_incomplete_data:
        critical_rules = """
### CRITICAL BILLING RULES - READ CAREFULLY:

1. **VECTORSTORE HAS INCOMPLETE DATA** - Some procedure codes have missing descriptions.
   CHECK THE PREVIOUS SUPERBILL MAPPING below for procedure name to CPT code mappings.

2. If the current clinical context has the SAME procedure names as previous superbill, 
   you MUST use the SAME CPT codes.

3. Match procedure names from clinical context to the CPT codes from previous superbill

4. Include appropriate ICD-10 diagnosis codes from the context

5. APPLY MODIFIERS according to the modifier rules below
"""
    else:
        critical_rules = """
### CRITICAL BILLING RULES - READ CAREFULLY:

1. **ONLY use CPT codes from the Allowed PROCEDURE CPT Rules list** - DO NOT invent codes

2. Match procedure names from clinical context to the allowed CPT codes

3. Include appropriate ICD-10 diagnosis codes from the context

4. APPLY MODIFIERS according to the modifier rules below
"""
    
    prompt = f"""
You are a medical billing expert for dermatology.
{critical_rules}
{prev_superbill_text}
### Suggested Modifiers for this Context
{json.dumps(possible_modifiers, indent=2)}

### MODIFIER RULES (from billing knowledge base):
{modifier_rules_text if modifier_rules_text else "No specific modifier rules retrieved"}

### KEY MODIFIER APPLICATION GUIDANCE:
- When E/M visit occurs WITH procedures on same day: Apply modifier 25 to E/M code
- When multiple DIFFERENT procedures at different sites: Apply modifier 59 for distinct procedural services
- When procedure has laterality (left/right): Apply appropriate LT/RT modifier
- When same procedure repeated: Apply repeat procedure modifier 76

### Procedures Identified in Context
{json.dumps(procedure_names, indent=2)}

### Diagnosis Context
{json.dumps(diagnosis_context, indent=2)}

### Allowed PROCEDURE CPT Rules (USE ONLY THESE CODES)
{json.dumps(procedure_rules, indent=2)}

### Allowed E/M Rules
{json.dumps(enm_rules, indent=2)}

### Patient Clinical Context
{context}

### REQUIRED OUTPUT (STRICT JSON)
Return a JSON object with:
- procedures: array of procedures with proCode (from allowed list), quantity, modifiers array (APPLY MODIFIERS per rules above), dxCodes array
- enm: object with code (from allowed E/M list), modifiers array (apply E/M modifiers when procedures are also billed), and dxCodes array (primary diagnosis codes for the visit)

Example format:
{{
  "procedures": [
    {{
      "proCode": "11100",
      "quantity": 1,
      "modifiers": [],
      "dxCodes": ["D48.5"]
    }},
    {{
      "proCode": "17110",
      "quantity": 1,
      "modifiers": ["59"],
      "dxCodes": ["L82.0"]
    }}
  ],
  "enm": {{
    "code": "99213",
    "modifiers": ["25"],
    "dxCodes": ["L82.0", "D48.5"]
  }}
}}
"""

    return {**state, "prompt": prompt}





async def billing_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    seen = set()
    all_docs = []
    retrieval_plan = state.get("retrieval_plan", [])

    # First, do semantic search based on retrieval plan
    for item in retrieval_plan:
        query = item.get("query", "")
        doc_type = item.get("type", "")
        
        if not query or not doc_type:
            continue
            
        # Retrieve documents based on semantic similarity
        docs = vectorstore.similarity_search(
            query=query,
            k=10,
            filter={"type": doc_type}
        )

        for doc in docs:
            key = (
                doc.metadata.get("proCode")
                or doc.metadata.get("modifier")
                or doc.metadata.get("enmCode")
            )

            if key and key not in seen:
                seen.add(key)
                all_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

    # Check if retrieved procedure docs have proper descriptions
    # If not, we'll need to use previous superbill as fallback
    has_incomplete_docs = False
    for doc in all_docs:
        meta = doc.get("metadata", {})
        if meta.get("type") == "procedure":
            code_desc = meta.get("codeDesc", "")
            if not code_desc or code_desc.strip() == "":
                has_incomplete_docs = True
                break

    # Store flag for prompt_node to use previous superbill when needed
    return {
        **state, 
        "retrieved_docs": all_docs,
        "has_incomplete_vectorstore_data": has_incomplete_docs
    }






async def retrieval_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_openai_llm()
    context = state.get("context", "")

    prompt = f"""
You are extracting retrieval intent for a medical billing system.

Analyze the clinical context carefully and extract:

1. procedure_names: Extract the EXACT procedure names mentioned in the structured data sections 
   (e.g., "Biopsy", "Destruction Benign", "Excision Malignant", "MOHS Surgery", etc.)
   Look for procedures in sections like [BIOPSY PROCEDURES], [GENERAL PROCEDURES], [MOHS PROCEDURES]

2. procedure_categories: High-level categories these procedures fall into
   (e.g., "biopsy", "destruction", "excision", "mohs", "repair", "injection")

3. enm_level: The E/M level if an evaluation occurred. Look at patient status (new vs established) 
   and complexity. Return as single string like "established patient level 3" or "new patient level 4"

4. possible_modifiers: Identify modifier codes that may apply based on the clinical scenario:
   - If E/M visit AND procedures on same day, include "25"
   - If multiple different procedure types at different sites, include "59"
   - If laterality documented (left/right), include "LT" or "RT"
   - If bilateral procedure, include "50"
   - If procedure repeated same day, include "76"
   - If professional component only, include "26"

5. diagnosis_context: Key diagnosis terms that affect billing (e.g., "benign", "malignant", "premalignant", 
   specific conditions like "seborrheic keratosis", "actinic keratosis", "melanoma")

Return STRICT JSON ONLY:
{{
  "procedure_names": [],
  "procedure_categories": [],
  "enm_level": "",
  "possible_modifiers": [],
  "diagnosis_context": []
}}

Clinical Context:
{context}
"""

    response = await llm.ainvoke([AIMessage(content=prompt)])

    try:
        intent = json.loads(response.content) #type: ignore
    except Exception:
        intent = {
            "procedure_names": [],
            "procedure_categories": [],
            "enm_level": "",
            "possible_modifiers": [],
            "diagnosis_context": []
        }

    # Safety normalization
    intent["procedure_names"] = intent.get("procedure_names", []) or []
    intent["procedure_categories"] = intent.get("procedure_categories", []) or []
    intent["enm_level"] = intent.get("enm_level", "") or ""
    intent["possible_modifiers"] = intent.get("possible_modifiers", []) or []
    intent["diagnosis_context"] = intent.get("diagnosis_context", []) or []

    return {**state, "retrieval_intent": intent}



def retrieval_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    intent = state.get("retrieval_intent", {})
    
    # Get previous superbill from cleaned_patient_data
    cleaned_patient_data = state.get("cleaned_patient_data", {})
    note_id = state.get("note_id")
    note_data = cleaned_patient_data.get(note_id, {}) if note_id else {}
    
    plan = []
    seen_queries = set()

    def add_query(query_type: str, query: str):
        """Add query if not duplicate"""
        key = (query_type, query.lower())
        if key not in seen_queries:
            seen_queries.add(key)
            plan.append({"type": query_type, "query": query})

    # Store previous superbill codes for potential fallback use
    previous_superbill = note_data.get("previous_superbill", [])
    previous_codes = set()
    previous_code_map = {}  # Maps procedure name to CPT code
    for item in previous_superbill:
        cpt = item.get("CPT", "")
        procedure_name = item.get("Procedure", "")
        if cpt:
            previous_codes.add(cpt)
            if procedure_name:
                previous_code_map[procedure_name.lower().strip()] = cpt

    # Build queries from exact procedure names (most specific)
    for proc_name in intent.get("procedure_names", []):
        add_query("procedure", f"{proc_name} CPT code billing rules")

    # Build queries from procedure categories  
    for category in intent.get("procedure_categories", []):
        add_query("procedure", f"Dermatology {category} CPT billing rules")

    # Build queries combining categories with diagnosis context for specificity
    diagnosis_terms = intent.get("diagnosis_context", [])
    for category in intent.get("procedure_categories", []):
        for dx_term in diagnosis_terms:
            add_query("procedure", f"{category} {dx_term} CPT code")

    # E/M level query
    enm_level = intent.get("enm_level", "")
    if enm_level:
        add_query("enm", f"{enm_level} E/M encounter billing rules")
    
    # Also add generic E/M queries based on patient type in context
    # This helps ensure we get E/M codes even if specific level wasn't extracted
    context = state.get("context", "").lower()
    if "new patient" in context:
        add_query("enm", "new patient office visit E/M code")
    if "established" in context or "followup" in context or "follow-up" in context or "f/u" in context:
        add_query("enm", "established patient office visit E/M code")

    # Modifier queries
    for mod in intent.get("possible_modifiers", []):
        add_query("modifier", f"modifier {mod} billing rules")
    
    # Store previous codes and mapping for potential fallback use by other nodes
    return {
        **state, 
        "retrieval_plan": plan, 
        "previous_superbill_codes": list(previous_codes),
        "previous_code_map": previous_code_map
    }



















async def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_openai_llm()
    prompt = state.get("prompt")

    if not prompt:
        raise ValueError("Missing prompt")

    response = await llm.ainvoke([AIMessage(content=prompt)])

    try:
        parsed = json.loads(response.content) #type: ignore
    except Exception as e:
        logger.error(f"Invalid LLM JSON: {response.content}")
        raise ValueError("LLM returned invalid JSON")

    return {**state, "llm_response": parsed}










def clinical_context_node(state: dict) -> dict:
    cleaned_patient_data = state.get("cleaned_patient_data", {})
    context = context_builder(cleaned_patient_data)
    return {**state, "context": context}


async def patient_data_node(state: dict) -> dict:
    prepared_state = await prepare_patient_data(state)
    return prepared_state




def billing_logic_node(state: dict) -> dict:
    llm_output = state.get("llm_response", {})
    retrieved_docs = state.get("retrieved_docs", [])
    has_incomplete_data = state.get("has_incomplete_vectorstore_data", False)
    
    # Get previous superbill from cleaned_patient_data
    cleaned_patient_data = state.get("cleaned_patient_data", {})
    note_id = state.get("note_id")
    note_data = cleaned_patient_data.get(note_id, {}) if note_id else {}
    previous_superbill = note_data.get("previous_superbill", [])

    # Build rules from retrieved docs
    rules = {}
    for d in retrieved_docs:
        meta = d.get("metadata", {})
        key = meta.get("proCode") or meta.get("enmCode")
        if key:
            rules[key] = meta

    # If vectorstore has incomplete data, add rules from previous superbill as fallback
    if has_incomplete_data:
        for item in previous_superbill:
            cpt = item.get("CPT", "")
            if cpt and cpt not in rules:
                # Create a minimal rule from previous superbill
                rules[cpt] = {
                    "proCode": cpt,
                    "codeDesc": item.get("Procedure", ""),
                    "ChargePerUnit": item.get("Charge Per Unit", "NO") == "YES",
                    "minQty": 1,
                    "maxQty": 10,  # Default max
                    "type": "procedure"
                }

    bill_items = []

    for proc in llm_output.get("procedures", []):
        code = proc["proCode"]
        rule = rules.get(code)
        if not rule:
            continue

        qty = proc.get("quantity", 1)
        min_qty = int(rule.get("minQty", 1) or 1)
        max_qty = int(rule.get("maxQty", 1) or 1)

        charge_per_unit = rule.get("ChargePerUnit", False)
        if not charge_per_unit:
            qty = 1
        else:
            qty = max(min(qty, max_qty), min_qty)

        bill_items.append({
            "code": code,
            "quantity": qty,
            "modifiers": proc.get("modifiers", []),
            "dxCodes": proc.get("dxCodes", []),
            "description": rule.get("codeDesc", ""),
            "chargePerUnit": charge_per_unit
        })

    enm = llm_output.get("enm", {})
    
    # Ensure E/M has all required fields
    if enm:
        enm_rule = rules.get(enm.get("code"), {})
        enm = {
            "code": enm.get("code"),
            "modifiers": enm.get("modifiers", []),
            "dxCodes": enm.get("dxCodes", []),
            "description": enm_rule.get("enmCodeDesc", "Evaluation & Management")
        }

    return {
        **state,
        "bill_items": bill_items,
        "enm": enm
    }



def html_render_node(state: dict) -> dict:
    """Renders the final billing HTML from the state."""
    final_bill = state.get("final_bill", "")
    return {**state, "final_response": final_bill}
    

async def test_agent(state: dict):
    """test the billing agent graph end-to-end"""
    cleaned_data = await patient_data_node(state)
    clinical_context = clinical_context_node(cleaned_data)
    print("Clinical Context:", clinical_context)


if __name__ == "__main__":
    import asyncio
    sample_state = {
        "note_id": 648734,
    }
    asyncio.run(test_agent(sample_state))
