# Security Policy

## Reporting

Please report vulnerabilities or accidental data/secret exposure privately via
[GitHub private vulnerability reporting](https://github.com/brindeshwar/medsignal/security/advisories/new)
rather than a public issue.

## Scope notes

- Secrets (`OPENFDA_API_KEY`, `HF_TOKEN`) live only in a local `.env`, which is git-ignored.
- Raw review datasets are not stored in this repo. If you find personal information in any committed file, report it and it will be removed.
- Model outputs are research artefacts and must not be used for clinical decisions.
