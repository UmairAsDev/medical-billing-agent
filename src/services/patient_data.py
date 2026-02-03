import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from sqlalchemy import text
from database.deps import async_db_session
from database.db import async_engine
from loguru import logger


logger.add("logs/data_preparation.log", rotation="10 MB")


async def get_notes(note_id: int):
    """Fetch a single note and attach biopsy, general, mohs, and prescription results."""
    async with async_db_session() as db:


        notes_query = text(
            """
        SELECT
            pn.noteId, pn.provider, pn.physician, pn.referringPhysician, pn.noteDate, pn.patientId,
            npn.complaints, npn.pastHistory, npn.assesment, npn.reviewofsystem, npn.currentmedication,
            npn.`procedure`, npn.biopsyNotes, npn.mohsNotes, npn.allergy, npn.examination, npn.patientSummary,
            group_concat(concat(dc.icd10Code, ' ', d.dxDescription)) AS diagnoses, pos.posName as PlaceOfService, CONCAT(p.firstName, ' ', p.lastName) as 'Rendering Provider', CONCAT(p2.firstName, ' ', p2.lastName) as 'Physician', CONCAT(p3.firstName, ' ', p3.lastName) as 'Referring Provider', CONCAT(p4.firstName, ' ', p4.lastName) as 'Billing Provider'
            FROM progressNotes pn
            LEFT JOIN providers p ON p.providerId = pn.provider 
            LEFT JOIN providers p2 ON p2.providerId = pn.physician 
            LEFT JOIN providers p3 ON p3.providerId = pn.referringPhysician
            LEFT JOIN providers p4 ON p4.providerId = pn.billingProvider 
            LEFT JOIN newProgressNotes npn ON pn.noteId = npn.noteId
            LEFT JOIN placeOfService pos ON pos.posCodes = pn.placeOfService 
            LEFT JOIN pnAssessment pa ON pa.noteId = pn.noteId
            LEFT JOIN diagnosis d ON d.dxId = pa.dxId
            LEFT JOIN diagnosisCodes dc ON dc.dxId = d.dxId AND dc.dxCodeId = pa.dxCodeId
            WHERE pn.physicianSignDate IS NOT NULL 
            AND pn.noteId = :note_id
        """
        )

        result = await db.execute(notes_query, {"note_id": note_id})
        notes = result.mappings().all()
        # logger.info(f"Notes result for note_id {note_id}: {notes}")
        if not notes:
            return {"message": f"No notes found for note_id {note_id}"}
        note_row = notes[0]

        data = {
            note_id: {
                "note": dict(note_row),
                "biopsy": [],
                "general": [],
                "mohs": [],
                "prescriptions": [],
                "previous_superbill": [],
            },
        }
        # logger.info(f"Initial data structure: {data}")
        patient_id = note_row.get("patientId")
        logger.info(f"Patient ID for note_id {note_id}: {patient_id}")
        if patient_id is None:
            logger.error(f"No patientId found for note_id {note_id}")
            return {"message": f"No patientId found for note_id {note_id}"}

        biopsy_query = text(
            """
            SELECT papb.noteId, pl.proName, papb.ruleOutDx, sl.site, sll.location, papb.lesionSize, papb.checkMargin, pml.`method`, pcl.cleansing, pal.anesthesia, pbl.blade,
        papb.undermining, papb.underminingSize, papb.woundSize, papb.closureSize, papb.intSuture, papb.intSutureSize, papb.extSuture, papb.extSutureSize,
        papb.dogEar, papb.dogEarCount, papb.flapClosure, pdl.dressing, papb.pathology, papb.pathRefPrefix, papb.pathRefNo, papb.frozenSection, papb.frozenRefPrefix,
        papb.frozenRefNo, papb.void, papb.cancel, papb.cancelBy, papb.cancelDetail, papb.notes, papb.lastModified, papb.delayedRepair, papb.marginOptions,
        papb.repairDate, papb.deleted
        FROM pnAssessmentProBiop papb
        LEFT JOIN proList pl ON pl.proId = papb.proId
        LEFT JOIN siteList sl ON sl.siteId = papb.siteId
        LEFT JOIN siteLocationList sll ON sll.locationId = papb.locationId
        LEFT JOIN proMethodList pml ON pml.methodId = papb.methodId
        LEFT JOIN proCleansingList pcl ON pcl.cleansingId = papb.cleansingId
        LEFT JOIN proAnesthesiaList pal ON pal.anesthesiaId = papb.anesthesiaId
        LEFT JOIN proBladeList pbl ON pbl.bladeId = papb.bladeId
        LEFT JOIN proDressingList pdl ON pdl.dressingId = papb.dressingId
            WHERE papb.noteId = :note_id
        """
        )
        bio_results = await db.execute(biopsy_query, {"note_id": note_id})
        biopsy_notes = bio_results.mappings().all()
        # if not biopsy_results:
        #     return {"message": f"No biopsy data found for note_id {note_id}"}
        data[note_id]["biopsy"] = [dict(row) for row in biopsy_notes]
        general_query = text(
            """
        SELECT
        papd.noteId, pl.proName, papd.proSeqNo, pml.`method`, pcl.choice, sl.site, sll.location,
        papd.qty, papd.billingSize, papd.drugId, pl.drugName, papd.drugUnit, papd.proSiteLocation
        FROM pnAssessmentProDet papd
        LEFT JOIN proList pl ON pl.proId = papd.proId
        LEFT JOIN pathMethodList pml ON pml.methodId = papd.methodId
        LEFT JOIN proChoiceList pcl ON pcl.choiceId = papd.choiceId
        LEFT JOIN siteList sl ON sl.siteId = papd.siteId
        LEFT JOIN siteLocationList sll ON sll.locationId = papd.locationId
        WHERE papd.noteId = :note_id
        """
        )
        general_results =await db.execute(general_query, {"note_id": note_id})
        general_notes = general_results.mappings().all()
        # if not general_results:
        #     return {"message": f"No general procedure data found for note_id {note_id}"}
        data[note_id]["general"] = [dict(row) for row in general_notes]

        mohs_query = text(
            """
        SELECT 
        papm.noteId, pl.proName , papm.assessmentId, papm.mohsId, papm.drawingId, papm.providerId, papm.asprin, papm.coumadin, papm.plavix, papm.vitamine,
        papm.warfarin, sl.site, sll.location, papm.preOpSize, pcl.cleansing, pal.anesthesia, papm.preMohsNote, papm.postMohsSizeLinear, papm.curettage, papm.hPressure,
        papm.hCautery, papm.hLigation, papm.hGelfoam, papm.hPacking, papm.mohsNote, papm.postMohsSize, papm.refForRepair, papm.electroDessi, papm.refRepairTo, papm.repairId,
        papm.undermining, papm.underminingSize, papm.intSuture, papm.intSutureSize, papm.extSuture, papm.extSutureSize, papm.dogEarRepaired, papm.noOfDogEars, papm.flapClosure, papm.postRepairSize,
        papm.repairNote, papm.dSiteId, papm.dDefectSize, papm.dLocationId, papm.dRepairId, papm.dUndermining, papm.dIntSuture, papm.dIntSutureSize, papm.dExtSuture, papm.dExtSutureSize,
        papm.dDogEarRepaired, papm.dNoOfDogEars, papm.dFlapClosure, papm.dPostRepairSize, papm.dNote, papm.aquaGuard, papm.bacitracin, papm.bactroban, papm.dryGauze, papm.centany,
        papm.neosporin, papm.opSite, papm.telfa, papm.steriStrips, papm.vaseline, papm.xeroform, papm.pressDressing, papm.polysporin, papm.woundSeal, papm.noComplication,
        papm.bleeding, papm.pain, papm.hypotension, papm.otherComplication, papm.finalNote, papm.delayedRepair, papm.repairDetailId, papm.deleted
        FROM pnAssessmentProMohs papm
        LEFT JOIN proList pl ON pl.proId = papm.proId
        LEFT JOIN siteList sl ON sl.siteId = papm.siteId
        LEFT JOIN siteLocationList sll ON sll.locationId = papm.locationId
        LEFT JOIN proCleansingList pcl ON pcl.cleansingId = papm.cleansingId
        LEFT JOIN proAnesthesiaList pal ON pal.anesthesiaId = papm.anesthesiaId
            WHERE papm.noteId = :note_id
        """
        )
        mohs_results = await db.execute(mohs_query, {"note_id": note_id})
        mohs_notes = mohs_results.mappings().all()
        data[note_id]["mohs"] = [dict(row) for row in mohs_notes]

        prescription_query = text(
            """
            SELECT * FROM erxPrescriptions WHERE progressnoteId = :note_id
        """
        )
        prescription_results = (
            await db.execute(prescription_query, {"note_id": note_id})
        )
        prescription_notes = prescription_results.mappings().all()
        logger.info(f"Prescription results: {prescription_notes}")
        data[note_id]["prescriptions"] = [dict(row) for row in prescription_notes]

        previous_medications_query = text(
            """
            SELECT * FROM progressNotes WHERE noteId < :note_id AND patientId = :patient_id and pathNote = 0 ORDER BY noteId DESC LIMIT 1
        """
        )
        previous_results = (
            await db.execute(
                previous_medications_query, {"note_id": note_id, "patient_id": patient_id}
            )
        )
        previous_notes = previous_results.mappings().all()
        data[note_id]["previous_medications"] = [dict(row) for row in previous_notes]
        prev_note_id = previous_notes[0]["noteId"] if previous_notes else None
        # if not prev_note_id:
        #     return {"message": f"No previous note found for note_id {note_id}"}

        previous_superbill_query = text(
            """SELECT CASE WHEN p.enmCodeId > 0 THEN ecl.enmCodeDesc ELSE pl.proName END as `Procedure`,
        CASE WHEN p.enmCodeId > 0 THEN ecl.enmCode ELSE pcl.proCode END as `CPT`,
        p.modifierId,
        CASE WHEN p.chargePerUnit = 1 THEN 'YES' ELSE 'NO' END as `Charge Per Unit`,
        p.qty AS Quantity,
        GROUP_CONCAT(dc.icd10Code SEPARATOR ', ')
        FROM pnSB p
        LEFT JOIN pnSBAssessment ps ON ps.sBId = p.sBId
        LEFT JOIN diagnosisCodes dc ON dc.dxId = ps.dxId AND dc.dxCodeId = ps.dxCodeId
        LEFT JOIN proCodeList pcl ON pcl.proCodeId = p.proCodeId
        LEFT JOIN proList pl ON pl.proId = p.proId
        LEFT JOIN enmCodeList ecl ON ecl.enmCodeId = p.enmCodeId
        WHERE noteId = :prev_note_id
        GROUP BY ps.sBId"""
        )
        previous_superbill_results = (
           await db.execute(previous_superbill_query, {"prev_note_id": prev_note_id})
        )
        previous_superbill_data = previous_superbill_results.mappings().all()
        # if not previous_superbill_results:
        #     return {"message": f"No previous superbill found for previous note_id {prev_note_id}"}
        data[note_id]["previous_superbill"] = [
            dict(row) for row in previous_superbill_data
        ]

        await db.commit()  # Commit the session
        return data




async def main(note_id:int):
    try:
        data = await get_notes(note_id)
        print(f"Data for note_id {note_id}: {data}")
        return data
    finally:
        await async_engine.dispose()



if __name__ == "__main__":
    note_id = 700726
    asyncio.run(main(note_id))
