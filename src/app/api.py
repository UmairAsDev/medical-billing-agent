from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger

from src.agent.graph import build_billing_graph

router = APIRouter()


@router.get("/superbill/{note_id}", response_class=HTMLResponse)
async def get_superbill(note_id: int):
    billing_graph = await build_billing_graph()

    result = await billing_graph.ainvoke({
        "note_id": note_id
    })

    logger.debug(f"Billing graph result for note_id {note_id}: {result}")

    html = result.get("final_response")

    if not html:
        raise HTTPException(
            status_code=404,
            detail="No billable services found for this encounter"
        )

    logger.info(f"Generated superbill for note_id {note_id}")
    return HTMLResponse(content=html)
