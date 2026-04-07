from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/media/imit-learn/ISR_2T3/HoverAI_2/RAG")
SOURCE_JSONL = ROOT / "skoltech_events_dataset_final_spaced.jsonl"
OUTPUT_JSONL = ROOT / "skoltech_events_dataset_verified.jsonl"
OUTPUT_TXT = ROOT / "skoltech_events_verified.txt"
FACT_CHECKED_AT = "2026-04-07"


FIXES = {
    "skoltech_event_0001": {
        "verification_status": "contradicted_fixed",
        "verification_notes": (
            "The original dataset had the wrong year and date. "
            "The official event page shows March 26, 2026, 11:00 a.m.-1:10 p.m., room B4-3006."
        ),
        "verified_source_url": "https://events.skoltech.ru/training_towards_excellence_in_teaching_assistantship",
        "language": "en",
        "text": (
            "Training session on March 26, 2026, from 11:00 a.m. to 1:10 p.m. at "
            "Bolshoy Boulevard 30, bld. 1, room B4-3006. The event focuses on teaching assistant skills."
        ),
        "summary": "TA training on March 26, 2026, 11:00-13:10, room B4-3006.",
        "source_url": "https://events.skoltech.ru/training_towards_excellence_in_teaching_assistantship",
        "source_domain": "events.skoltech.ru",
        "source_type": "official_webpage",
        "metadata": {
            "date": "2026-03-26",
            "time": "11:00-13:10",
            "room": "B4-3006",
            "venue": "Bolshoy Boulevard 30, bld. 1",
        },
    },
    "skoltech_event_0002": {
        "verification_status": "confirmed",
        "verification_notes": "Confirmed by the official event page.",
        "verified_source_url": "https://events.skoltech.ru/commencement2025",
        "text": "Graduation ceremony on June 27, 2025, at 16:00 in the Skoltech Main Hall.",
        "summary": "Commencement on June 27, 2025, at 16:00 in the Main Hall.",
        "metadata": {
            "date": "2025-06-27",
            "time": "16:00",
            "room": "Main Hall",
        },
    },
    "skoltech_event_0003": {
        "verification_status": "confirmed",
        "verification_notes": "Confirmed by the official Skoltech news page.",
        "verified_source_url": "https://www.skoltech.ru/en/news/breakthrough-formula-frontiers-progress-outcome",
        "language": "en",
        "text": "Frontiers of Progress conference held on May 22-23, 2025, at the Skoltech campus.",
        "summary": "Frontiers of Progress conference, May 22-23, 2025.",
        "metadata": {
            "date": "2025-05-22 to 2025-05-23",
            "room": "Not specified",
            "venue": "Skoltech campus",
        },
    },
    "skoltech_event_0004": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event exists, but the raw dataset used a generic source URL and sparse text.",
        "verified_source_url": "https://events.skoltech.ru/ls_winter_school",
        "source_url": "https://events.skoltech.ru/ls_winter_school",
        "source_domain": "events.skoltech.ru",
        "text": "Life Sciences Winter School on February 14-16, 2025, held offline at the Skoltech campus.",
        "summary": "Life Sciences Winter School, February 14-16, 2025, offline at Skoltech.",
        "metadata": {
            "date": "2025-02-14 to 2025-02-16",
            "room": "Not specified",
            "venue": "Skoltech campus",
            "format": "offline",
        },
    },
    "skoltech_event_0005": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event title is listed on the official portal, but detailed page data was not captured.",
        "verified_source_url": "https://events.skoltech.ru/",
        "source_url": "https://events.skoltech.ru/",
        "source_domain": "events.skoltech.ru",
        "text": "ICPC Moscow Regional Contest listed on the official Skoltech events portal for November 23, 2025.",
        "summary": "ICPC Moscow Regional Contest, November 23, 2025.",
        "metadata": {
            "date": "2025-11-23",
            "room": "Not specified",
        },
    },
    "skoltech_event_0006": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event title is listed on the official portal, but detailed page data was not captured.",
        "verified_source_url": "https://events.skoltech.ru/",
        "source_url": "https://events.skoltech.ru/",
        "source_domain": "events.skoltech.ru",
        "language": "ru",
        "text": "Event listed on the official Skoltech events portal for December 4, 2025.",
        "summary": "December industrial event listed for December 4, 2025.",
        "metadata": {
            "date": "2025-12-04",
            "room": "Not specified",
        },
    },
    "skoltech_event_0007": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event title is listed on the official portal, but detailed page data was not captured.",
        "verified_source_url": "https://events.skoltech.ru/",
        "source_url": "https://events.skoltech.ru/",
        "source_domain": "events.skoltech.ru",
        "language": "en",
        "text": "IT-Purple event listed on the official Skoltech events portal for March 15, 2025.",
        "summary": "IT-Purple, March 15, 2025.",
        "metadata": {
            "date": "2025-03-15",
            "room": "Not specified",
        },
    },
    "skoltech_event_0008": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event title is listed on the official portal, but detailed page data was not captured.",
        "verified_source_url": "https://events.skoltech.ru/",
        "source_url": "https://events.skoltech.ru/",
        "source_domain": "events.skoltech.ru",
        "language": "ru",
        "text": "Mental arithmetic championship listed on the official Skoltech events portal for May 11, 2025.",
        "summary": "Mental arithmetic championship, May 11, 2025.",
        "metadata": {
            "date": "2025-05-11",
            "room": "Not specified",
        },
    },
    "skoltech_event_0009": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event title is listed on the official portal, but detailed page data was not captured.",
        "verified_source_url": "https://events.skoltech.ru/",
        "source_url": "https://events.skoltech.ru/",
        "source_domain": "events.skoltech.ru",
        "language": "en",
        "text": "Self-made woman event listed on the official Skoltech events portal for April 22, 2025.",
        "summary": "Self-made woman, April 22, 2025.",
        "metadata": {
            "date": "2025-04-22",
            "room": "Not specified",
        },
    },
    "skoltech_event_0010": {
        "verification_status": "confirmed",
        "verification_notes": "Confirmed by the official Selection Days page.",
        "verified_source_url": "https://events.skoltech.ru/selection_days_2025",
        "text": "Selection Days 2025 admissions event held on July 24-25, 2025.",
        "summary": "Selection Days 2025, July 24-25, 2025.",
        "source_url": "https://events.skoltech.ru/selection_days_2025",
        "source_domain": "events.skoltech.ru",
        "metadata": {
            "date": "2025-07-24 to 2025-07-25",
            "room": "Not specified",
        },
    },
    "skoltech_event_0011": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The event exists, but the original dataset pointed to a secondary news page rather than the main event site.",
        "verified_source_url": "https://smiles.skoltech.ru/",
        "source_url": "https://smiles.skoltech.ru/",
        "source_domain": "smiles.skoltech.ru",
        "language": "en",
        "text": (
            "SMILES Summer School of Machine Learning 2025 was held on July 14-27, 2025, "
            "at Harbin Institute of Technology in China."
        ),
        "summary": "SMILES 2025, July 14-27, 2025, Harbin Institute of Technology, China.",
        "metadata": {
            "date": "2025-07-14 to 2025-07-27",
            "room": "Not specified",
            "venue": "Harbin Institute of Technology, China",
        },
    },
    "skoltech_event_0012": {
        "verification_status": "partially_confirmed",
        "verification_notes": "Confirmed in the academic calendar, but not as a detailed event page.",
        "verified_source_url": "https://www.skoltech.ru/en/academic-calendar/2025-2026",
        "source_url": "https://www.skoltech.ru/en/academic-calendar/2025-2026",
        "source_domain": "www.skoltech.ru",
        "text": "Onboarding course for MSc-1 scheduled for August 18-29, 2025, according to the academic calendar.",
        "summary": "MSc-1 onboarding course, August 18-29, 2025.",
        "metadata": {
            "date": "2025-08-18 to 2025-08-29",
            "room": "Not specified",
        },
    },
    "skoltech_event_0013": {
        "verification_status": "partially_confirmed",
        "verification_notes": "Confirmed by the Selection Days page, but the raw dataset pointed to the wrong source URL.",
        "verified_source_url": "https://events.skoltech.ru/selection_days_2025",
        "source_url": "https://events.skoltech.ru/selection_days_2025",
        "source_domain": "events.skoltech.ru",
        "language": "en",
        "text": (
            "Math paper exam for Data Science and Advanced Computational Sciences applicants "
            "scheduled for July 25, 2025, from 10:00 to 13:00 in room E-B4-3007."
        ),
        "summary": "Math paper exam, July 25, 2025, 10:00-13:00, room E-B4-3007.",
        "metadata": {
            "date": "2025-07-25",
            "time": "10:00-13:00",
            "room": "E-B4-3007",
        },
    },
    "skoltech_event_0014": {
        "verification_status": "partially_confirmed",
        "verification_notes": "Confirmed by the official Industry Day site, but the raw dataset was too generic.",
        "verified_source_url": "https://industryday.skoltech.ru/english",
        "source_url": "https://industryday.skoltech.ru/english",
        "source_domain": "industryday.skoltech.ru",
        "text": "Industry Day official event confirmed for October 15, 2025.",
        "summary": "Industry Day, October 15, 2025.",
        "metadata": {
            "date": "2025-10-15",
            "room": "Not specified",
        },
    },
    "skoltech_event_0015": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The lecture exists in the Lecture Hub archive for 2025, but the exact date was not captured in this pass.",
        "verified_source_url": "https://www.skoltech.ru/en/lecture-hub-archive",
        "source_url": "https://www.skoltech.ru/en/lecture-hub-archive",
        "source_domain": "www.skoltech.ru",
        "language": "en",
        "text": "Dmitrii Kriukov lecture on computational biology of aging listed in the official Lecture Hub archive for 2025.",
        "summary": "Dmitrii Kriukov lecture listed in the 2025 Lecture Hub archive.",
        "metadata": {
            "date": "2025",
            "room": "Not specified",
        },
    },
    "skoltech_event_0016": {
        "verification_status": "partially_confirmed",
        "verification_notes": "The official news page confirms the item, but the captured date is the publication date, not necessarily the event date.",
        "verified_source_url": "https://www.skoltech.ru/en/news/skoltech-host-geographical-dictation",
        "source_url": "https://www.skoltech.ru/en/news/skoltech-host-geographical-dictation",
        "source_domain": "www.skoltech.ru",
        "text": "Skoltech hosted Geographical dictation; the official news item was published on November 18, 2025.",
        "summary": "Geographical dictation, news published on November 18, 2025.",
        "metadata": {
            "date": "2025-11-18",
            "date_kind": "publication_date",
            "room": "Not specified",
        },
    },
    "skoltech_event_0017": {
        "verification_status": "contradicted_fixed",
        "verification_notes": "The original dataset used the wrong year. The official page corresponds to November 25, 2023.",
        "verified_source_url": "https://events.skoltech.ru/weekend_light",
        "source_url": "https://events.skoltech.ru/weekend_light",
        "source_domain": "events.skoltech.ru",
        "language": "ru",
        "text": "Weekend with light event scheduled for November 25, 2023, according to the official event page.",
        "summary": "Weekend with light, November 25, 2023.",
        "metadata": {
            "date": "2023-11-25",
            "room": "Not specified",
        },
    },
    "skoltech_event_0018": {
        "verification_status": "partially_confirmed",
        "verification_notes": "Confirmed in the academic calendar, but not as a dedicated detailed event page.",
        "verified_source_url": "https://www.skoltech.ru/en/academic-calendar/2025-2026",
        "source_url": "https://www.skoltech.ru/en/academic-calendar/2025-2026",
        "source_domain": "www.skoltech.ru",
        "text": "Onboarding Q&A for PhD-1 scheduled for September 1, 2025, according to the academic calendar.",
        "summary": "PhD-1 onboarding Q&A, September 1, 2025.",
        "metadata": {
            "date": "2025-09-01",
            "room": "Not specified",
        },
    },
}


def merge_record(record: dict, updates: dict) -> dict:
    result = deepcopy(record)
    for key, value in updates.items():
        if key == "metadata":
            merged_metadata = dict(result.get("metadata") or {})
            merged_metadata.update(value)
            result["metadata"] = merged_metadata
        else:
            result[key] = value
    result["fact_checked_at"] = FACT_CHECKED_AT
    return result


def main() -> None:
    records = []
    with SOURCE_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record = merge_record(record, FIXES.get(record["id"], {
                "verification_status": "unverified",
                "verification_notes": "No fact-check data applied.",
                "verified_source_url": record.get("source_url"),
            }))
            records.append(record)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n\n")

    with OUTPUT_TXT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n\n")

    print(f"Wrote {len(records)} records to {OUTPUT_JSONL}")
    print(f"Wrote {len(records)} records to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
