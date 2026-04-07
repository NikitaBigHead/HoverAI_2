# Fact-check report for `skoltech_events_dataset_final_spaced.jsonl`

Date checked: 2026-04-07

Status legend:
- `confirmed`: the core fact is confirmed by an official source.
- `partially_confirmed`: the event exists, but the dataset is incomplete, too generic, or points to a weak/wrong source URL.
- `contradicted`: the dataset contains a factual mismatch against the official source.
- `unverified`: no reliable confirmation found during this pass.

## Results

| ID | Title | Status | Notes | Official source |
|---|---|---|---|---|
| `skoltech_event_0001` | Training: Towards excellence in Teaching Assistantship | contradicted | Dataset says `2025-10-21`, but the official event page shows `March 26, 2026`, `11:00 a.m.-1:10 p.m.`, `B4-3006`, `Bolshoy Boulevard 30, bld. 1`. | https://events.skoltech.ru/training_towards_excellence_in_teaching_assistantship |
| `skoltech_event_0002` | Commencement 2025 | confirmed | Official page confirms `June 27, 2025`, `16:00`, `Skoltech Main Hall`. | https://events.skoltech.ru/commencement2025 |
| `skoltech_event_0003` | Frontiers of Progress Conference | confirmed | Official news confirms the conference took place on `May 22-23, 2025` at Skoltech. | https://www.skoltech.ru/en/news/breakthrough-formula-frontiers-progress-outcome |
| `skoltech_event_0004` | Life Sciences Winter School | partially_confirmed | Event exists; official page shows `February 14-16, 2025`, offline, Skoltech campus. Dataset only says `2025` and uses a generic events URL. | https://events.skoltech.ru/ls_winter_school |
| `skoltech_event_0005` | ICPC Moscow Regional Contest | partially_confirmed | Event title appears on the official Skoltech events portal with date `23.11.2025`, but the dataset has only generic text and a generic source URL. | https://events.skoltech.ru/ |
| `skoltech_event_0006` | Декабрьский слет производственников | partially_confirmed | Event title appears on the official Skoltech events portal with date `04.12.2025`, but the dataset has only generic text and a generic source URL. | https://events.skoltech.ru/ |
| `skoltech_event_0007` | IT-Purple | partially_confirmed | Event title appears on the official Skoltech events portal with date `15.03.2025`, but the dataset has only generic text and a generic source URL. | https://events.skoltech.ru/ |
| `skoltech_event_0008` | Чемпионат по ментальной арифметике | partially_confirmed | Event title appears on the official Skoltech events portal with date `11.05.2025`, but the dataset has only generic text and a generic source URL. | https://events.skoltech.ru/ |
| `skoltech_event_0009` | Self-made woman | partially_confirmed | Event title appears on the official Skoltech events portal with date `22.04.2025`, but the dataset has only generic text and a generic source URL. | https://events.skoltech.ru/ |
| `skoltech_event_0010` | Selection Days 2025 | confirmed | Official page confirms `July 24-25, 2025`. | https://events.skoltech.ru/selection_days_2025 |
| `skoltech_event_0011` | SMILES Summer School of Machine Learning 2025 | partially_confirmed | Event exists. Official SMILES page shows `July 14-27, 2025`, at Harbin Institute of Technology in China. Dataset is too generic and its source URL is not the main event page. | https://smiles.skoltech.ru/ |
| `skoltech_event_0012` | Onboarding course for MSc-1 | partially_confirmed | Official academic calendar confirms `August 18-29, 2025`. Dataset only says `2025` and uses a generic events URL. | https://www.skoltech.ru/en/academic-calendar/2025-2026 |
| `skoltech_event_0013` | Math paper exam for Data Science | partially_confirmed | Official Selection Days page confirms `July 25, 2025`, `10:00-13:00`, `E-B4-3007` for Data Science and ACS applicants. Dataset is too generic and points to the wrong source URL. | https://events.skoltech.ru/selection_days_2025 |
| `skoltech_event_0014` | Industry Day | partially_confirmed | Official sources confirm Industry Day `15.10.2025`. Dataset is too generic and uses a weak source URL. | https://industryday.skoltech.ru/english |
| `skoltech_event_0015` | Dmitrii Kriukov Lecture: Computational biology of aging | partially_confirmed | Lecture exists in the official Lecture Hub archive for `2025`, but the dataset lacks date/details and points to a generic events URL. | https://www.skoltech.ru/en/lecture-hub-archive |
| `skoltech_event_0016` | Geographical dictation | partially_confirmed | Official Skoltech news confirms a `Geographical dictation` item published on `November 18, 2025`. Dataset is too generic and lacks the actual date/details. | https://www.skoltech.ru/en/news/skoltech-host-geographical-dictation |
| `skoltech_event_0017` | Выходной со светом | contradicted | Official page exists, but it is for `November 25, 2023`, not `2025`. Dataset year is therefore wrong. | https://events.skoltech.ru/weekend_light |
| `skoltech_event_0018` | Onboarding Q&A for PhD-1 | partially_confirmed | Official academic calendar confirms `September 1, 2025`. Dataset only says `2025` and uses a generic events URL. | https://www.skoltech.ru/en/academic-calendar/2025-2026 |

## Summary

- Total records checked: `18`
- `confirmed`: `4`
- `partially_confirmed`: `12`
- `contradicted`: `2`
- `unverified`: `0`

## Main dataset issues

1. Many records use the generic URL `https://skoltech.ru/en/events` instead of a stable event-specific source.
2. Many records contain placeholder-like text such as `IT event`, `biology lecture`, `industrial meeting`, or just `2025` instead of exact metadata.
3. At least two records contain factual errors:
   - `Training: Towards excellence in Teaching Assistantship`
   - `Выходной со светом`
4. Image links are not evidence for the event itself; they should not be treated as proof during ingestion.

## Recommendation

This dataset is acceptable as `raw ingestion v0`, but it is **not clean enough to be treated as a trusted factual dataset**. Before production use, each record should be normalized to:
- exact `source_url`
- exact `date`
- exact `time` if available
- exact `venue/room` if available
- richer `text` extracted from the official page
- separate `verification_status` field
