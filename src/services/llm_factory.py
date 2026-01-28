import sys
import boto3
from pathlib import Path
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import settings
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()



bedrock_runtime = boto3.client(
    settings.BEDROCK_SERVICE_NAME,
    region_name=settings.AWS_REGION,
)




def get_llm():
    """
    Factory pattern to return the configured LLM.
    Changes based on .env configuration.
    """
    model_kwargs = {"temperature": 0.0}
    llm = ChatBedrock(
        client=boto3.client("bedrock-runtime", region_name=settings.AWS_REGION),
        model=settings.MODEL_ID,
        model_kwargs=model_kwargs,
        streaming=True 
    ) #type: ignore
    return llm


OpenAI(api_key=settings.OPENAI_API_KEY) 

def get_openai_llm():
    """
    Factory pattern to return the configured OpenAI LLM.
    Changes based on .env configuration.
    """
    
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL_ID,
        temperature=0.0,
        streaming=True 
    ) 
    return llm





if __name__ == "__main__":
    llm = get_openai_llm()
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain medical billing in simple terms.")
    ])
    print(response)