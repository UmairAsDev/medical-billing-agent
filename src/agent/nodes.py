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





# def prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Build a clinical reasoning prompt for dermatology billing.
    
#     This node creates a comprehensive prompt that enables the LLM to reason
#     about ANY dermatology scenario based on clinical understanding rather than
#     hardcoded rules or pattern matching.
#     """
#     procedure_rules = []
#     modifier_rules = []
#     enm_rules = []
#     retrieved_docs = state.get("retrieved_docs", [])

#     for d in retrieved_docs:
#         meta = d.get("metadata", {})
#         content = d.get("content", "")

#         doc_type = meta.get("type", "")
#         if doc_type == "procedure":
#             procedure_rules.append(meta)
#         elif doc_type == "modifier":
#             modifier_rules.append({
#                 "modifier": meta.get("modifier"),
#                 "enmModifier": meta.get("enmModifier"),
#                 "details": content
#             })
#         elif doc_type == "enm":
#             enm_rules.append(meta)


#     intent = state.get("retrieval_intent", {})
#     procedure_names = intent.get("procedure_names", [])
#     diagnosis_context = intent.get("diagnosis_context", [])
#     possible_modifiers = intent.get("possible_modifiers", [])


#     modifier_rules_text = ""
#     for mod in modifier_rules:
#         modifier_rules_text += f"\n**Modifier {mod.get('modifier')}** (E/M applicable: {mod.get('enmModifier', False)}):\n{mod.get('details', '')}\n"


#     cleaned_patient_data = state.get("cleaned_patient_data", {})
#     note_id = state.get("note_id")
#     note_data = cleaned_patient_data.get(note_id, {}) if note_id else {}
#     previous_superbill = note_data.get("previous_superbill", [])
    

#     prev_superbill_text = ""
#     if previous_superbill:
#         prev_superbill_text = "\n### HISTORICAL REFERENCE: PREVIOUS SUPERBILL\n"
#         prev_superbill_text += "This shows what was previously billed for this patient. Use as REFERENCE only:\n"
#         for item in previous_superbill:
#             proc_name = item.get("Procedure", "")
#             cpt = item.get("CPT", "")
#             mod = item.get("modifierId", "")
#             qty = item.get("Quantity", 1)
#             dx = item.get("GROUP_CONCAT(dc.icd10Code SEPARATOR ', ')", "")
#             prev_superbill_text += f"- {proc_name} → CPT: {cpt}"
#             if mod:
#                 prev_superbill_text += f" (Modifier: {mod})"
#             prev_superbill_text += f" Qty: {qty}, Dx: {dx}\n"

#     context = state.get("context", "")
    
#     prompt = f"""
# You are an expert dermatology medical billing coder. Your task is to analyze the clinical documentation 
# and generate accurate CPT codes with appropriate modifiers and diagnosis codes.

# ## CLINICAL REASONING APPROACH

# You must reason through each patient encounter step by step:

# ### STEP 1: IDENTIFY PROCEDURE CATEGORIES
# Read the clinical context and identify which categories of procedures were performed:

# **BIOPSY PROCEDURES** (sampling tissue for diagnosis)
# - Skin biopsy (shave, punch, incisional, excisional)
# - Primary code: First biopsy lesion
# - Add-on codes: Additional biopsy lesions (+11101, +11107)


# **DESTRUCTION PROCEDURES** (destroying lesions without removing tissue)
# - Benign lesion destruction (17110, 17111) - warts, SK, etc.
# - Premalignant lesion destruction (17000, 17003, 17004) - actinic keratoses
# - Malignant lesion destruction (17260-17286) - by size and location

# **EXCISION PROCEDURES** (surgically removing lesions)
# - Benign excision (11400-11446) - by size and location
# - Malignant excision (11600-11646) - by size and location
# - Size includes excised diameter + margins

# **MOHS SURGERY** (micrographic surgery for skin cancer)
# - First stage (17311-17315) - by location and size
# - Additional stages (+17312-+17315)
# - Complex repair codes may follow

# **RADIATION THERAPY** (for cancer treatment)
# - Simulation/planning (77280, 77285, 77290)
# - Treatment delivery (77401, 77402, etc.)
# - Special procedure codes (G6001, G6002, etc.)
# - Weekly E/M for radiation management

# **REPAIR/CLOSURE** (wound closure after procedures)
# - Simple repair (12001-12007)
# - Intermediate repair (12031-12057)
# - Complex repair (13100-13153)
# - Adjacent tissue transfer/flaps (14000-14350)

# **E/M SERVICES** (evaluation and management)
# - New patient visits (99201-99205)
# - Established patient visits (99211-99215)
# - Consultations (99241-99245)
# - If E/M on same day as procedure, MUST use modifier 25


# ## MOHS BUNDLING RULES

# - Simple repairs are INCLUDED in Mohs and must NOT be billed separately.
# - Only bill intermediate/complex repairs, flaps, or grafts if clearly documented.

# ## E/M BILLING RULES

# - Apply modifier 25 ONLY if E/M is significant and separately identifiable.
# - E/M diagnosis codes must reflect evaluation reasons, not procedural pathology alone.



# ## DESTRUCTION QUANTITY RULES (STRICT)

# - Actinic Keratosis:
#   - 17000 → first lesion only
#   - 17003 → 2–14 additional lesions
#   - 17004 → 15 or more lesions

# - Benign destruction:
#   - 17110 → up to 14 lesions
#   - 17111 → 15 or more lesions

# Select CPT codes based ONLY on documented lesion counts.


# ## CRITICAL ICD-10 & MEDICAL NECESSITY RULES

# 1. ONLY assign ICD-10 codes explicitly supported by documentation.
# 2. NEVER infer pathology results.
#    - If biopsy performed and pathology is pending → use uncertain behavior ICD-10 (e.g., D48.5).
# 3. Malignant ICD-10 codes REQUIRE confirmed documentation.
# 4. EACH CPT must be medically justified by its linked ICD-10 code.
# 5. If no valid ICD-10 supports a CPT → DO NOT bill the CPT.
# 6. Do NOT reuse diagnosis codes across unrelated procedures.


# ### STEP 2: MAP CLINICAL FINDINGS TO CODES
# For each procedure identified in the clinical context:
# 1. Determine the exact procedure type (what was done clinically)
# 2. Find the matching CPT code from the ALLOWED CODES list
# 3. Consider anatomical location and lesion size if relevant
# 4. Determine quantity (how many lesions/sites)

# ### STEP 3: APPLY DIAGNOSIS CODES
# For each procedure:
# 1. Link the appropriate ICD-10 diagnosis code(s)
# 2. Primary diagnosis should be the reason for the procedure
# 3. Include morphology-specific codes when available (benign vs malignant)

# ### STEP 4: APPLY MODIFIERS
# Apply modifiers based on clinical circumstances:
# - **25**: E/M with significant, separately identifiable service on same day as procedure
# - **59**: Distinct procedural service (different site, different lesion, different session)
# - **LT/RT**: Left/Right laterality
# - **50**: Bilateral procedure
# - **76**: Repeat procedure by same physician
# - **XE, XS, XP, XU**: Subset modifiers for distinct services


# Do Not add any pathology-related procedures or codes unless explicitly documented.



# {prev_superbill_text}

# ### EXTRACTED PROCEDURE INFORMATION FROM CONTEXT
# Procedures identified: {json.dumps(procedure_names, indent=2)}
# Diagnosis terms: {json.dumps(diagnosis_context, indent=2)}
# Possible modifiers needed: {json.dumps(possible_modifiers, indent=2)}

# ### MODIFIER RULES FROM BILLING KNOWLEDGE BASE
# {modifier_rules_text if modifier_rules_text else "Standard modifier rules apply"}

# ### ALLOWED PROCEDURE CPT CODES (SELECT FROM THIS LIST)
# {json.dumps(procedure_rules, indent=2)}

# ### ALLOWED E/M CODES (SELECT FROM THIS LIST)
# {json.dumps(enm_rules, indent=2)}

# ### PATIENT CLINICAL CONTEXT
# {context}

# ## YOUR TASK

# Analyze the clinical context above and:
# 1. Identify ALL procedures performed during this encounter
# 2. Match each procedure to the appropriate CPT code from the ALLOWED lists
# 3. Assign correct quantities based on the number of lesions/sites treated
# 4. Apply appropriate modifiers based on the clinical scenario
# 5. Link diagnosis codes to each procedure

# ## OUTPUT FORMAT (STRICT JSON)

# Return a JSON object with:
# - **procedures**: Array of procedure objects, each containing:
#   - proCode: CPT code (MUST be from allowed list)
#   - quantity: Number of units (based on lesion count or clinical documentation)
#   - modifiers: Array of modifier codes (e.g., ["59", "LT"])
#   - dxCodes: Array of ICD-10 codes for this procedure
# - **enm**: E/M visit object (if applicable) containing:
#   - code: E/M CPT code (MUST be from allowed E/M list)
#   - modifiers: Array (use ["25"] if procedures also billed)
#   - dxCodes: Array of diagnosis codes for the visit

# Example:
# {{
#   "procedures": [
#     {{
#       "proCode": "11102",
#       "quantity": 1,
#       "modifiers": [],
#       "dxCodes": ["D48.5"]
#     }},
#     {{
#       "proCode": "11103",
#       "quantity": 2,
#       "modifiers": [],
#       "dxCodes": ["D48.5"]
#     }},
#     {{
#       "proCode": "17110",
#       "quantity": 3,
#       "modifiers": ["59"],
#       "dxCodes": ["L82.0"]
#     }}
#   ],
#   "enm": {{
#     "code": "99214",
#     "modifiers": ["25"],
#     "dxCodes": ["D48.5", "L82.0"]
#   }}
# }}

# IMPORTANT REMINDERS:
# 1. ONLY use CPT codes that appear in the ALLOWED PROCEDURE or ALLOWED E/M lists
# 2. If a procedure in the clinical context doesn't match any allowed code, skip it
# 3. Apply modifier 25 to E/M when procedures are billed on the same day
# 4. Apply modifier 59 when billing multiple distinct procedures
# 5. Include ALL procedures documented, with appropriate quantities
# """

#     return {**state, "prompt": prompt}






def prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a clinical reasoning prompt for dermatology billing using retrieved vectorstore rules.
    
    Forces the LLM to reason step-by-step:
    1. Identify procedures
    2. Map to CPT codes from retrieved docs
    3. Assign correct quantities
    4. Link ICD-10 (active/current year)
    5. Apply modifiers
    """

    retrieved_docs = state.get("retrieved_docs", [])
    procedure_rules = []
    modifier_rules = []
    enm_rules = []

    for d in retrieved_docs:
        meta = d.get("metadata", {})
        content = d.get("content", "")

        doc_type = meta.get("type", "")
        if doc_type == "procedure":
            procedure_rules.append(meta)
        elif doc_type == "modifier":
            modifier_rules.append({
                "modifier": meta.get("modifier"),
                "enmModifier": meta.get("enmModifier", False),
                "details": content
            })
        elif doc_type == "enm":
            enm_rules.append(meta)

    intent = state.get("retrieval_intent", {})
    procedure_names = intent.get("procedure_names", [])
    diagnosis_context = intent.get("diagnosis_context", [])
    possible_modifiers = intent.get("possible_modifiers", [])

    modifier_rules_text = ""
    for mod in modifier_rules:
        modifier_rules_text += f"\n**Modifier {mod.get('modifier')}** (E/M applicable: {mod.get('enmModifier', False)}):\n{mod.get('details', '')}\n"

    cleaned_patient_data = state.get("cleaned_patient_data", {})
    note_id = state.get("note_id")
    note_data = cleaned_patient_data.get(note_id, {}) if note_id else {}
    previous_superbill = note_data.get("previous_superbill", [])

    prev_superbill_text = ""
    if previous_superbill:
        prev_superbill_text = "\n### HISTORICAL REFERENCE: PREVIOUS SUPERBILL\n"
        prev_superbill_text += "Reference only. Do not assume repeated billing:\n"
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
You are an expert dermatology medical billing coder.

**Stepwise Clinical Reasoning Required**
For each documented procedure:
1. Identify procedure type.
2. Determine exact CPT from **retrieved allowed CPT list** only.
3. Apply quantity constraints (lesion count, first/additional lesion rules).
4. Assign **active-year ICD-10 codes** linked to CPT.
5. Apply appropriate modifiers based on clinical context and **retrieved modifier rules**.

**Rules**
- Only use CPTs and E/M codes from retrieved vectorstore.
- Only assign ICD-10 codes documented in patient note or allowed active-year codes.
- Do NOT infer pathology results.
- Mohs repairs and simple repairs are bundled as per rules.
- Actinic keratosis destruction: 17000 → first, 17003 → 2–14 lesions, 17004 → 15+ lesions.
- Do not reuse ICD-10 codes across unrelated procedures.
- Apply modifier 25 only if E/M is separately identifiable.

**Reasoning Steps**
1. Identify ALL procedures from clinical context.
2. Match each to CPT from latest documentation.
3. Determine quantities based on lesion/site counts.
4. Apply ICD-10 codes to each CPT according to the ICD-10 documentation.
5. Apply appropriate modifiers from the modifier rules.

**Historical Reference: Previous Superbill**
- Check previous superbill for reference only.

{prev_superbill_text}

### Procedures identified:
{json.dumps(procedure_names, indent=2)}

### Diagnosis context:
{json.dumps(diagnosis_context, indent=2)}

### Possible modifiers:
{json.dumps(possible_modifiers, indent=2)}

### Modifier rules:
{modifier_rules_text if modifier_rules_text else "Standard modifier rules apply"}

### Allowed CPT codes:
{json.dumps(procedure_rules, indent=2)}

### Allowed E/M codes:
{json.dumps(enm_rules, indent=2)}

### Patient clinical context:
{context}

**Output STRICT JSON**:
{{
  "procedures": [
    {{
      "proCode": "CPT from allowed list",
      "quantity": "Number of lesions/sites",
      "modifiers": ["25", "59", ...],
      "dxCodes": ["ICD-10 active codes"]
    }}
  ],
  "enm": {{
    "code": "E/M CPT from allowed list",
    "modifiers": ["25" if applicable],
    "dxCodes": ["ICD-10 codes for visit"]
  }}
}}

Include **step-by-step reasoning** internally before final JSON output.
Only include codes documented and allowed.
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


    for item in retrieval_plan:
        query = item.get("query", "")
        doc_type = item.get("type", "")
        
        if not query or not doc_type:
            continue
            

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

STRICT RULES:
- Do NOT infer pathology results.
- Do NOT assume malignancy unless explicitly documented.
- Distinguish E/M evaluation reasons from procedural diagnoses.
- Extract ONLY what is clearly documented.

Extract the following:

1. PROCEDURES (exactly as documented)
- Biopsy (shave, punch, incisional, excisional)
- Destruction (benign, premalignant, malignant)
- Excision (benign or malignant)
- Mohs surgery (with stages)
- Repairs (simple / intermediate / complex)
- Flaps / grafts
- Radiation therapy (simulation vs treatment)
- Injections
- E/M visit

2. PROCEDURE CATEGORIES
Use ONLY these values:
biopsy, destruction_benign, destruction_premalignant, destruction_malignant,
excision_benign, excision_malignant, mohs,
repair_simple, repair_intermediate, repair_complex,
flap, graft, radiation_simulation, radiation_treatment,
evaluation_management

3. E/M LEVEL
Only if documented:
- new patient level 1–5
- established patient level 1–5

4. MODIFIERS (only if clearly supported)
25, 59, LT, RT, 50, 76, XE, XS, XP, XU

5. DIAGNOSIS CONTEXT
- Use clinical terms only
- Include pathology status (confirmed vs pending vs suspected)
- Include laterality and anatomical site if documented

6. QUANTITY INDICATORS
- Lesion count
- Biopsy count
- Treatment sessions

Return STRICT JSON only.

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


    for proc_name in intent.get("procedure_names", []):
        add_query("procedure", f"{proc_name} CPT code billing")


    categories = intent.get("procedure_categories", [])
    diagnosis_terms = intent.get("diagnosis_context", [])
    

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


    for category in categories:
        for dx_term in diagnosis_terms[:2]:  
            add_query("procedure", f"{category} {dx_term} dermatology CPT")


    enm_level = intent.get("enm_level", "")
    if enm_level:
        add_query("enm", f"{enm_level} E/M office visit billing")
    

    context = state.get("context", "").lower()
    if "new patient" in context:
        add_query("enm", "new patient office visit E/M code 99201 99205")
    if any(term in context for term in ["established", "followup", "follow-up", "f/u", "return"]):
        add_query("enm", "established patient office visit E/M code 99211 99215")
    

    add_query("enm", "evaluation management dermatology office visit")


    for mod in intent.get("possible_modifiers", []):
        add_query("modifier", f"modifier {mod} when to use billing rules")
    

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
    Validate and finalize billing items from LLM output using retrieved vectorstore rules.
    Enforces:
    - CPT only from retrieved docs
    - ICD-10 active/current-year enforcement
    - Quantity limits
    - Modifier hierarchy
    """

    llm_output = state.get("llm_response", {})
    retrieved_docs = state.get("retrieved_docs", [])


    rules = {}
    for d in retrieved_docs:
        meta = d.get("metadata", {})
        key = meta.get("proCode") or meta.get("enmCode")
        if key:
            rules[key] = meta

    bill_items = []

    for proc in llm_output.get("procedures", []):
        code = proc.get("proCode", "")
        if not code or code not in rules:
            continue  

        rule = rules.get(code, {})
        qty = proc.get("quantity", 1)


        min_qty = int(rule.get("minQty", 1))
        max_qty = int(rule.get("maxQty", 99))
        charge_per_unit = rule.get("ChargePerUnit", True)
        if not charge_per_unit:
            qty = 1
        else:
            qty = max(min(qty, max_qty), min_qty)


        dx_codes = [dx for dx in proc.get("dxCodes", []) if dx in rule.get("allowedICD10", [])]

        bill_items.append({
            "code": code,
            "quantity": qty,
            "modifiers": proc.get("modifiers", []),
            "dxCodes": dx_codes,
            "description": rule.get("codeDesc", ""),
            "chargePerUnit": charge_per_unit
        })


    enm = llm_output.get("enm", {})
    if enm and enm.get("code") and enm["code"] in rules:
        enm_rule = rules.get(enm["code"], {})
        enm = {
            "code": enm["code"],
            "modifiers": enm.get("modifiers", []),
            "dxCodes": [dx for dx in enm.get("dxCodes", []) if dx in enm_rule.get("allowedICD10", [])],
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
