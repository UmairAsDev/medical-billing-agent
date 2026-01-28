def html_render_node(state: dict) -> dict:
    bill_items = state.get("bill_items", [])
    enm = state.get("enm")
    retrieved_docs = state.get("retrieved_docs", [])
    
    # Build lookup for procedure descriptions and metadata from retrieved docs
    proc_info = {}
    enm_info = {}
    for doc in retrieved_docs:
        meta = doc.get("metadata", {})
        if meta.get("type") == "procedure":
            code = meta.get("proCode")
            if code:
                proc_info[code] = {
                    "description": meta.get("codeDesc", "Procedure"),
                    "chargePerUnit": meta.get("ChargePerUnit", False)
                }
        elif meta.get("type") == "enm":
            code = meta.get("enmCode")
            if code:
                enm_info[code] = {
                    "description": meta.get("enmCodeDesc", "Evaluation & Management"),
                }

    rows = []

    # Procedure rows first (matching your image format)
    for item in bill_items:
        code = item["code"]
        info = proc_info.get(code, {})
        desc = info.get("description", "Procedure")
        per_unit = "Yes" if info.get("chargePerUnit", False) else "No"
        
        rows.append(f"""
        <tr>
            <td>{desc}</td>
            <td>{code}</td>
            <td>{", ".join(item.get("modifiers", [])) or "-"}</td>
            <td>{", ".join(item.get("dxCodes", [])) or "-"}</td>
            <td>{item["quantity"]}</td>
            <td>{per_unit}</td>
        </tr>
        """)

    # E/M row last (matching your image format)
    if enm and enm.get("code"):
        enm_code = enm["code"]
        enm_desc = enm_info.get(enm_code, {}).get("description", "Office visit")
        enm_dx = enm.get("dxCodes", [])
        
        rows.append(f"""
        <tr>
            <td>{enm_desc}</td>
            <td>{enm_code}</td>
            <td>{", ".join(enm.get("modifiers", [])) or "-"}</td>
            <td>{", ".join(enm_dx) if enm_dx else "-"}</td>
            <td>1</td>
            <td>Yes</td>
        </tr>
        """)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Billing Summary</title>
<style>
body {{
    font-family: Arial, sans-serif;
    background: #ffffff;
    color: #000;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}
th, td {{
    border: 1px solid #444;
    padding: 8px;
    font-size: 13px;
    text-align: left;
}}
th {{
    background-color: #f0f0f0;
    font-weight: bold;
}}
.header {{
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}}
.subheader {{
    font-size: 13px;
    margin-bottom: 20px;
}}
</style>
</head>

<body>

<div class="header">Medical Billing Summary</div>
<div class="subheader">
    Note ID: {state.get("note_id", "N/A")}
</div>

<table>
<thead>
<tr>
    <th>Procedures</th>
    <th>Code</th>
    <th>Modifier</th>
    <th>Dx Code</th>
    <th>Qty.</th>
    <th>Per Unit</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>

</body>
</html>
"""

    return {**state, "final_response": html}
