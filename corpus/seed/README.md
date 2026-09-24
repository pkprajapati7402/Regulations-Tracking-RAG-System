# Seed corpus

These eight files are **offline fixtures**: RBI-format circulars written to mirror the
structure, numbering, tabular layout, cross-references and supersession language of real
RBI Master Circulars / Master Directions. They exist so the system (and the test-suite and
the evaluation harness) is reproducible in a sandbox with no network access, and so the
amendment case-study always has a working chain:

| File | Role |
|---|---|
| `01-kyc-master-direction-2016.txt` | Base document (KYC Master Direction, para 6 = periodic updation) |
| `02-kyc-amendment-2026.txt` | **Amends** para 6 of the above (`in supersession of ...`) — drives the stale-data case study |
| `03-digital-lending-guidelines.txt` | Cooling-off period, LSP fees, data privacy |
| `04-prudential-framework-stressed-assets.txt` | SMA classification table, resolution timelines |
| `05-master-circular-customer-service.txt` | Compensation table for failed transactions |
| `06-master-direction-credit-debit-cards.txt` | Card closure penalty, activation OTP |
| `07-master-circular-priority-sector-lending.txt` | PSL targets table |
| `08-cyber-security-framework-banks.txt` | Incident reporting timeline, CISO requirements |

**They are not a substitute for the primary source.** To index the genuine documents, run:

```bash
python -m ingestion.fetch_rbi --category master-circulars --limit 40 --out corpus/raw
python -m ingestion.reindex --path corpus/raw
```

which downloads the real circulars from rbi.org.in into `corpus/raw/` (git-ignored) and
indexes them exactly the same way.
