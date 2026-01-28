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
from langchain_core.messages import AIMessage, HumanMessage
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
    """
    Build a clinical reasoning prompt for dermatology billing.
    
    This node creates a comprehensive prompt that enables the LLM to reason
    about ANY dermatology scenario based on clinical understanding rather than
    hardcoded rules or pattern matching.
    """
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
            modifier_rules.append({
                "modifier": meta.get("modifier"),
                "enmModifier": meta.get("enmModifier"),
                "details": content
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

    # Get previous superbill as historical reference
    cleaned_patient_data = state.get("cleaned_patient_data", {})
    note_id = state.get("note_id")
    note_data = cleaned_patient_data.get(note_id, {}) if note_id else {}
    previous_superbill = note_data.get("previous_superbill", [])
    
    # Build previous superbill reference
    prev_superbill_text = ""
    if previous_superbill:
        prev_superbill_text = "\n### HISTORICAL REFERENCE: PREVIOUS SUPERBILL\n"
        prev_superbill_text += "This shows what was previously billed for this patient. Use as REFERENCE only:\n"
        for item in previous_superbill:
            proc_name = item.get("Procedure", "")
            cpt = item.get("CPT", "")
            mod = item.get("modifierId", "")
            qty = item.get("Quantity", 1)
            dx = item.get("GROUP_CONCAT(dc.icd10Code SEPARATOR ', ')", "")
            prev_superbill_text += f"- {proc_name} → CPT: {cpt}"
            if mod:
                prev_superbill_text += f" (Modifier: {mod})"
            prev_superbill_text += f" Qty: {qty}, Dx: {dx}\n"

    context = state.get("context", "")
    
    prompt = f"""
You are an expert dermatology medical billing coder. Your task is to analyze the clinical documentation 
and generate accurate CPT codes with appropriate modifiers and diagnosis codes.

## CLINICAL REASONING APPROACH

You must reason through each patient encounter step by step:

### STEP 1: IDENTIFY PROCEDURE CATEGORIES
Read the clinical context and identify which categories of procedures were performed:

**BIOPSY PROCEDURES** (sampling tissue for diagnosis)
- Skin biopsy (shave, punch, incisional, excisional)
- Primary code: First biopsy lesion
- Add-on codes: Additional biopsy lesions (+11101, +11107)

**DESTRUCTION PROCEDURES** (destroying lesions without removing tissue)
- Benign lesion destruction (17110, 17111) - warts, SK, etc.
- Premalignant lesion destruction (17000, 17003, 17004) - actinic keratoses
- Malignant lesion destruction (17260-17286) - by size and location

**EXCISION PROCEDURES** (surgically removing lesions)
- Benign excision (11400-11446) - by size and location
- Malignant excision (11600-11646) - by size and location
- Size includes excised diameter + margins

**MOHS SURGERY** (micrographic surgery for skin cancer)
- First stage (17311-17315) - by location and size
- Additional stages (+17312-+17315)
- Complex repair codes may follow

**RADIATION THERAPY** (for cancer treatment)
- Simulation/planning (77280, 77285, 77290)
- Treatment delivery (77401, 77402, etc.)
- Special procedure codes (G6001, G6002, etc.)
- Weekly E/M for radiation management

**REPAIR/CLOSURE** (wound closure after procedures)
- Simple repair (12001-12007)
- Intermediate repair (12031-12057)
- Complex repair (13100-13153)
- Adjacent tissue transfer/flaps (14000-14350)

**E/M SERVICES** (evaluation and management)
- New patient visits (99201-99205)
- Established patient visits (99211-99215)
- Consultations (99241-99245)
- If E/M on same day as procedure, MUST use modifier 25

### STEP 2: MAP CLINICAL FINDINGS TO CODES
For each procedure identified in the clinical context:
1. Determine the exact procedure type (what was done clinically)
2. Find the matching CPT code from the ALLOWED CODES list
3. Consider anatomical location and lesion size if relevant
4. Determine quantity (how many lesions/sites)

### STEP 3: APPLY DIAGNOSIS CODES
For each procedure:
1. Link the appropriate ICD-10 diagnosis code(s)
2. Primary diagnosis should be the reason for the procedure
3. Include morphology-specific codes when available (benign vs malignant)

### STEP 4: APPLY MODIFIERS
Apply modifiers based on clinical circumstances:
- **25**: E/M with significant, separately identifiable service on same day as procedure
- **59**: Distinct procedural service (different site, different lesion, different session)
- **LT/RT**: Left/Right laterality
- **50**: Bilateral procedure
- **76**: Repeat procedure by same physician
- **XE, XS, XP, XU**: Subset modifiers for distinct services

{prev_superbill_text}

### EXTRACTED PROCEDURE INFORMATION FROM CONTEXT
Procedures identified: {json.dumps(procedure_names, indent=2)}
Diagnosis terms: {json.dumps(diagnosis_context, indent=2)}
Possible modifiers needed: {json.dumps(possible_modifiers, indent=2)}

### MODIFIER RULES FROM BILLING KNOWLEDGE BASE
{modifier_rules_text if modifier_rules_text else "Standard modifier rules apply"}

### ALLOWED PROCEDURE CPT CODES (SELECT FROM THIS LIST)
{json.dumps(procedure_rules, indent=2)}

### ALLOWED E/M CODES (SELECT FROM THIS LIST)
{json.dumps(enm_rules, indent=2)}

### PATIENT CLINICAL CONTEXT
{context}

## YOUR TASK

Analyze the clinical context above and:
1. Identify ALL procedures performed during this encounter
2. Match each procedure to the appropriate CPT code from the ALLOWED lists
3. Assign correct quantities based on the number of lesions/sites treated
4. Apply appropriate modifiers based on the clinical scenario
5. Link diagnosis codes to each procedure

## OUTPUT FORMAT (STRICT JSON)

Return a JSON object with:
- **procedures**: Array of procedure objects, each containing:
  - proCode: CPT code (MUST be from allowed list)
  - quantity: Number of units (based on lesion count or clinical documentation)
  - modifiers: Array of modifier codes (e.g., ["59", "LT"])
  - dxCodes: Array of ICD-10 codes for this procedure
- **enm**: E/M visit object (if applicable) containing:
  - code: E/M CPT code (MUST be from allowed E/M list)
  - modifiers: Array (use ["25"] if procedures also billed)
  - dxCodes: Array of diagnosis codes for the visit

Example:
{{
  "procedures": [
    {{
      "proCode": "11102",
      "quantity": 1,
      "modifiers": [],
      "dxCodes": ["D48.5"]
    }},
    {{
      "proCode": "11103",
      "quantity": 2,
      "modifiers": [],
      "dxCodes": ["D48.5"]
    }},
    {{
      "proCode": "17110",
      "quantity": 3,
      "modifiers": ["59"],
      "dxCodes": ["L82.0"]
    }}
  ],
  "enm": {{
    "code": "99214",
    "modifiers": ["25"],
    "dxCodes": ["D48.5", "L82.0"]
  }}
}}

IMPORTANT REMINDERS:
1. ONLY use CPT codes that appear in the ALLOWED PROCEDURE or ALLOWED E/M lists
2. If a procedure in the clinical context doesn't match any allowed code, skip it
3. Apply modifier 25 to E/M when procedures are billed on the same day
4. Apply modifier 59 when billing multiple distinct procedures
5. Include ALL procedures documented, with appropriate quantities
"""

    return {**state, "prompt": prompt}





async def billing_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant billing codes and rules from the vectorstore.
    
    Uses semantic search to find CPT codes, E/M codes, and modifier rules
    based on the retrieval plan generated from clinical context.
    """
    seen = set()
    all_docs = []
    retrieval_plan = state.get("retrieval_plan", [])

    # Do semantic search based on retrieval plan
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

    return {**state, "retrieved_docs": all_docs}






async def retrieval_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract retrieval intent from clinical context for comprehensive dermatology billing.
    
    This node analyzes the clinical documentation to identify:
    - All procedures performed (biopsy, destruction, excision, mohs, radiation, etc.)
    - Diagnosis context (benign vs malignant, specific conditions)
    - E/M service level
    - Modifier applicability
    """
    llm = get_openai_llm()
    context = state.get("context", "")

    prompt = f"""
You are extracting billing intent from dermatology clinical documentation.

Analyze the clinical context THOROUGHLY and extract ALL relevant information:

## 1. PROCEDURE NAMES
Extract EVERY procedure mentioned in the clinical documentation:
- Look in structured sections like [BIOPSY PROCEDURES], [GENERAL PROCEDURES], [MOHS PROCEDURES], [RADIATION THERAPY]
- Look for procedure descriptions in note text
- Include: biopsies, destructions, excisions, mohs surgery, repairs, injections, radiation treatments

Common dermatology procedures to look for:
- Biopsy (shave, punch, incisional, excisional)
- Destruction Benign (warts, seborrheic keratosis, skin tags)
- Destruction Premalignant (actinic keratosis, AK)
- Destruction Malignant
- Excision Benign / Excision Malignant (by size)
- MOHS Micrographic Surgery (stages)
- Simple/Intermediate/Complex Repair
- Flap/Graft procedures
- Radiation therapy (simulation, treatment delivery)
- SRT (Superficial Radiation Therapy)
- E/M visits

## 2. PROCEDURE CATEGORIES
Classify procedures into billing categories:
- biopsy
- destruction_benign
- destruction_premalignant  
- destruction_malignant
- excision_benign
- excision_malignant
- mohs
- repair_simple
- repair_intermediate
- repair_complex
- flap
- graft
- radiation_simulation
- radiation_treatment
- injection
- evaluation_management

## 3. E/M LEVEL
Determine the E/M service level if an office visit occurred:
- Patient status: new vs established
- Visit complexity: level 1-5
- Format as: "established patient level 3" or "new patient level 4"

## 4. MODIFIERS NEEDED
Identify which modifiers should apply:
- 25: E/M visit WITH procedures on same day (significant, separately identifiable)
- 59: Distinct procedural service (different site, different lesion type, different session)
- LT/RT: Left or Right laterality
- 50: Bilateral procedure
- 76: Repeat procedure same physician same day
- XE: Separate encounter
- XS: Separate structure
- XP: Separate practitioner
- XU: Unusual non-overlapping service

## 5. DIAGNOSIS CONTEXT
Extract diagnosis-related terms:
- Benign vs Malignant vs Premalignant
- Specific conditions (seborrheic keratosis, actinic keratosis, basal cell carcinoma, melanoma, etc.)
- ICD-10 codes if mentioned
- Anatomical locations

## 6. QUANTITY INDICATORS
Look for quantity information:
- Number of lesions treated
- Number of biopsies performed
- Number of destructions
- Radiation treatment sessions/fractions

Return STRICT JSON:
{{
  "procedure_names": ["list of exact procedure names from the document"],
  "procedure_categories": ["list of billing categories"],
  "enm_level": "patient status and level",
  "possible_modifiers": ["list of applicable modifiers"],
  "diagnosis_context": ["list of diagnosis-related terms"],
  "quantity_info": {{"procedure_type": count}}
}}

Clinical Context:
{context}
"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    try:
        intent = json.loads(response.content) #type: ignore
    except Exception:
        intent = {
            "procedure_names": [],
            "procedure_categories": [],
            "enm_level": "",
            "possible_modifiers": [],
            "diagnosis_context": [],
            "quantity_info": {}
        }

    # Safety normalization
    intent["procedure_names"] = intent.get("procedure_names", []) or []
    intent["procedure_categories"] = intent.get("procedure_categories", []) or []
    intent["enm_level"] = intent.get("enm_level", "") or ""
    intent["possible_modifiers"] = intent.get("possible_modifiers", []) or []
    intent["diagnosis_context"] = intent.get("diagnosis_context", []) or []
    intent["quantity_info"] = intent.get("quantity_info", {}) or {}

    return {**state, "retrieval_intent": intent}



def retrieval_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive retrieval queries for dermatology billing codes.
    
    Creates queries to retrieve relevant CPT codes, E/M codes, and modifier rules
    from the vectorstore based on the clinical context.
    """
    intent = state.get("retrieval_intent", {})
    
    plan = []
    seen_queries = set()

    def add_query(query_type: str, query: str):
        """Add query if not duplicate"""
        key = (query_type, query.lower())
        if key not in seen_queries:
            seen_queries.add(key)
            plan.append({"type": query_type, "query": query})

    # Build queries from exact procedure names (most specific)
    for proc_name in intent.get("procedure_names", []):
        add_query("procedure", f"{proc_name} CPT code billing")

    # Build queries from procedure categories with diagnosis context
    categories = intent.get("procedure_categories", [])
    diagnosis_terms = intent.get("diagnosis_context", [])
    
    # Category-specific queries
    category_query_map = {
        "biopsy": ["skin biopsy CPT code", "shave biopsy", "punch biopsy", "excisional biopsy"],
        "destruction_benign": ["destruction benign lesion CPT", "17110 17111 destruction"],
        "destruction_premalignant": ["destruction premalignant lesion CPT", "actinic keratosis destruction"],
        "destruction_malignant": ["destruction malignant lesion CPT", "malignant destruction by size"],
        "excision_benign": ["excision benign lesion CPT", "benign excision by size"],
        "excision_malignant": ["excision malignant lesion CPT", "malignant excision margins"],
        "mohs": ["mohs micrographic surgery CPT", "mohs first stage", "mohs additional stage"],
        "repair_simple": ["simple repair wound closure CPT"],
        "repair_intermediate": ["intermediate repair CPT"],
        "repair_complex": ["complex repair CPT"],
        "flap": ["adjacent tissue transfer flap CPT"],
        "graft": ["skin graft CPT"],
        "radiation_simulation": ["radiation simulation planning CPT 77280"],
        "radiation_treatment": ["radiation treatment delivery CPT", "superficial radiation therapy SRT"],
        "injection": ["injection lesion CPT"],
    }
    
    for category in categories:
        if category in category_query_map:
            for query in category_query_map[category]:
                add_query("procedure", query)
        else:
            add_query("procedure", f"Dermatology {category} CPT billing")

    # Combine categories with diagnosis context for specificity
    for category in categories:
        for dx_term in diagnosis_terms[:3]:  # Limit to avoid too many queries
            add_query("procedure", f"{category} {dx_term} CPT code")

    # E/M level query
    enm_level = intent.get("enm_level", "")
    if enm_level:
        add_query("enm", f"{enm_level} E/M office visit billing")
    
    # Also add generic E/M queries based on common patterns
    context = state.get("context", "").lower()
    if "new patient" in context:
        add_query("enm", "new patient office visit E/M code 99201 99205")
    if any(term in context for term in ["established", "followup", "follow-up", "f/u", "return"]):
        add_query("enm", "established patient office visit E/M code 99211 99215")
    
    # Always get common E/M codes
    add_query("enm", "evaluation management dermatology office visit")

    # Modifier queries based on identified modifiers
    for mod in intent.get("possible_modifiers", []):
        add_query("modifier", f"modifier {mod} when to use billing rules")
    
    # Always include common modifier rules
    common_modifiers = ["25", "59"]
    for mod in common_modifiers:
        if mod not in intent.get("possible_modifiers", []):
            add_query("modifier", f"modifier {mod} billing rules")
    
    return {**state, "retrieval_plan": plan}



















async def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_openai_llm()
    prompt = state.get("prompt")

    if not prompt:
        raise ValueError("Missing prompt")

    response = await llm.ainvoke([HumanMessage(content=prompt)])

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
    """
    Validate and finalize billing items from LLM response.
    
    This node:
    1. Validates CPT codes against retrieved rules
    2. Applies quantity constraints (ChargePerUnit, min/max)
    3. Preserves modifiers and diagnosis codes from LLM
    """
    llm_output = state.get("llm_response", {})
    retrieved_docs = state.get("retrieved_docs", [])

    # Build rules from retrieved docs
    rules = {}
    for d in retrieved_docs:
        meta = d.get("metadata", {})
        key = meta.get("proCode") or meta.get("enmCode")
        if key:
            rules[key] = meta

    bill_items = []

    for proc in llm_output.get("procedures", []):
        code = proc.get("proCode", "")
        if not code:
            continue
            
        rule = rules.get(code)
        
        # Get quantity from LLM response
        qty = proc.get("quantity", 1)
        
        # Apply quantity constraints from rules if available
        if rule:
            min_qty = int(rule.get("minQty", 1) or 1)
            max_qty = int(rule.get("maxQty", 99) or 99)
            charge_per_unit = rule.get("ChargePerUnit", False)
            
            if not charge_per_unit:
                qty = 1
            else:
                qty = max(min(qty, max_qty), min_qty)
            
            description = rule.get("codeDesc", "")
        else:
            # No rule found, but still include if LLM returned it
            # The prompt instructed LLM to only use allowed codes
            charge_per_unit = True  # Default to allowing quantity
            description = ""

        bill_items.append({
            "code": code,
            "quantity": qty,
            "modifiers": proc.get("modifiers", []),
            "dxCodes": proc.get("dxCodes", []),
            "description": description,
            "chargePerUnit": charge_per_unit
        })

    enm = llm_output.get("enm", {})
    
    # Ensure E/M has all required fields
    if enm and enm.get("code"):
        enm_rule = rules.get(enm.get("code"), {})
        enm = {
            "code": enm.get("code"),
            "modifiers": enm.get("modifiers", []),
            "dxCodes": enm.get("dxCodes", []),
            "description": enm_rule.get("enmCodeDesc", "Evaluation & Management")
        }
    else:
        enm = {}

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
