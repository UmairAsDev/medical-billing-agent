import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
from openai import OpenAI
from config.gunicorn_conf import settings
load_dotenv()


OpenAI(api_key=settings.OPENAI_API_KEY)

DATA_DIR = "./data"
CHROMA_DIR = "./vectordb/chroma"
COLLECTION_NAME = "billing_knowledge"

PRO_CODE_FILE = "proCodeList.csv"
MODIFIER_FILE = "modifierList.csv"
ENM_FILE = "enmCodeList.csv"

def ingest_data():
    """Insert thye data into Chroma vector store."""
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )


    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    documents = []
    metadatas = []
    ids = []


    def safe(val):
        return "" if pd.isna(val) else str(val)


    pro_df = pd.read_csv(os.path.join(DATA_DIR, PRO_CODE_FILE))

    for _, row in pro_df.iterrows():
        text = f"""
    Procedure Code: {safe(row.proCode)}
    Description: {safe(row.codeDesc)}

    Rules:
    - Minimum Quantity: {safe(row.minQty)}
    - Maximum Quantity: {safe(row.maxQty)}
    - Minimum Size: {safe(row.minSize)}
    - Maximum Size: {safe(row.maxSize)}
    - Add-on Code: {safe(row.addOn)}
    - Associated CPT: {safe(row.associatedWithProCode)}
    - Charge Per Unit: {safe(row.chargePerUnit)}
    - billWithIntEM: {safe(row.billWithIntEM)}
    - billWithFUEM: {safe(row.billWithFUEM)}
    - leftRightSepration: {safe(row.leftRightSepration)}
    - billAlone: {safe(row.billAlone)}
    - splitInMultipleVisits: {safe(row.splitInMultipleVisits)}
    """

        documents.append(text.strip())
        metadatas.append({
            "type": "procedure",
            "proCode": safe(row.proCode),
            "codeDesc": safe(row.codeDesc),  # Added procedure description
            "addOn": bool(row.addOn) if not pd.isna(row.addOn) else False,
            "minQty": safe(row.minQty),
            "maxQty": safe(row.maxQty),
            "active": not bool(row.deleted) if "deleted" in row else True,
            "ChargePerUnit": bool(row.chargePerUnit) if not pd.isna(row.chargePerUnit) else False,
            "billWithIntEM": bool(row.billWithIntEM) if not pd.isna(row.billWithIntEM) else False,
            "billWithFUEM": bool(row.billWithFUEM) if not pd.isna(row.billWithFUEM) else False,
            "leftRightSepration": bool(row.leftRightSepration) if not pd.isna(row.leftRightSepration) else False,
            "billAlone": bool(row.billAlone) if not pd.isna(row.billAlone) else False,
            "splitInMultipleVisits": bool(row.splitInMultipleVisits) if not pd.isna(row.splitInMultipleVisits) else False,
        })
        ids.append(f"procedure_{row.proCode}_{uuid4()}")




    mod_df = pd.read_csv(os.path.join(DATA_DIR, MODIFIER_FILE))

    for _, row in mod_df.iterrows():
        text = f"""
    Modifier: {safe(row.modifier)}
    Short Description: {safe(row.modifierDesc)}

    Detailed Rule:
    {safe(row.modifierDetDesc)}

    Applicable to E/M: {safe(row.enmModifier)}
    """

        documents.append(text.strip())
        metadatas.append({
            "type": "modifier",
            "modifier": safe(row.modifier),
            "enmModifier": bool(row.enmModifier),
            "active": not bool(row.deleted) if "deleted" in row else True
        })
        ids.append(f"modifier_{row.modifier}_{uuid4()}")


    enm_df = pd.read_csv(os.path.join(DATA_DIR, ENM_FILE))

    for _, row in enm_df.iterrows():
        text = f"""
    ENM Details:{safe(row.enmCodeDesc)}
    E/M Code: {safe(row.enmCode)}
    Encounter Type: {safe(row.enmType)}
    Level: {safe(row.enmLevel)}
    

    Typical Encounter Time: {safe(row.encounterTime)} minutes
    Facility Code: {safe(row.facilityCode)}
    """

        documents.append(text.strip())
        metadatas.append({
            "type": "enm",
            "enmCodeDesc": safe(row.enmCodeDesc),
            "enmCode": safe(row.enmCode),
            "enmLevel": safe(row.enmLevel),
            "enmType": safe(row.enmType),
            "facilityCode": safe(row.facilityCode),
            "active": not bool(row.deleted) if "deleted" in row else True
        })
        ids.append(f"enm_{row.enmCode}_{uuid4()}")


    vectorstore.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("  Chroma ingestion completed successfully")
    print(f"   Total records indexed: {len(documents)}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Persisted at: {CHROMA_DIR}")






if __name__ == "__main__":
    ingest_data()