import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.app.schemas import BillingState, BillingAgentState

from langgraph.graph import StateGraph, END
from src.agent.nodes import (
    patient_data_node,
    clinical_context_node,
    retrieval_intent_node,
    retrieval_plan_node,
    billing_retriever_node,
    prompt_node,
    llm_node,
    billing_logic_node
)
from src.app.renderers import html_render_node




async def serialized_retrieval_node(state: dict) -> dict:
    """
    Execute retrieval queries serially to accumulate billing codes.
    
    This node runs each retrieval query from the plan and deduplicates
    the results to build a comprehensive set of allowed billing codes.
    """
    retrieval_plan = state.get("retrieval_plan", [])
    
    if not retrieval_plan:
        return {**state, "retrieved_docs": []}
    
    seen = set()
    all_docs = []

    for item in retrieval_plan:
        result = await billing_retriever_node({
            **state,
            "retrieval_plan": [item]
        })

        for doc in result.get("retrieved_docs", []):
            meta = doc.get("metadata", {})
            key = (
                meta.get("type"),
                meta.get("proCode")
                or meta.get("modifier")
                or meta.get("enmCode")
            )

            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    return {**state, "retrieved_docs": all_docs}




async def build_billing_graph():
    graph = StateGraph(BillingAgentState)

    graph.add_node("patient_data", patient_data_node)                 #type: ignore
    graph.add_node("clinical_context", clinical_context_node)         #type: ignore
    graph.add_node("retrieval_intent", retrieval_intent_node)         #type: ignore
    graph.add_node("retrieval_plan", retrieval_plan_node)             #type: ignore
    graph.add_node("retrieve_billing_knowledge", serialized_retrieval_node) #type: ignore
    graph.add_node("prompt_builder", prompt_node)                     #type: ignore
    graph.add_node("llm_decision", llm_node)                           #type: ignore
    graph.add_node("billing_logic", billing_logic_node)               #type: ignore

    graph.set_entry_point("patient_data")

    graph.add_edge("patient_data", "clinical_context")
    graph.add_edge("clinical_context", "retrieval_intent")
    graph.add_edge("retrieval_intent", "retrieval_plan")
    graph.add_edge("retrieval_plan", "retrieve_billing_knowledge")
    graph.add_edge("retrieve_billing_knowledge", "prompt_builder")
    graph.add_edge("prompt_builder", "llm_decision")
    graph.add_edge("llm_decision", "billing_logic")
    graph.add_node("html_render", html_render_node) #type: ignore
    graph.add_edge("billing_logic", "html_render")
    graph.add_edge("html_render", END)

    return graph.compile()




# if __name__ == "__main__":
#     import asyncio
    
#     async def main():
#         billing_graph = await build_billing_graph()
#         result = await billing_graph.ainvoke({
#             "note_id": 704073
#         })
#         print(result)
    
#     asyncio.run(main())



