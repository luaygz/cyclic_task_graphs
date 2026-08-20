# Owner release checklist

- [ ] Replace `LICENSE` with the owner-selected project license.
- [ ] Replace placeholder author, paper title, and repository URL fields in
      `CITATION.cff`.
- [ ] Confirm the TextCraft and Finance Agent attributions with counsel or the
      institution's release policy.
- [ ] Revoke or rotate tracked browser sessions from the private repository's
      history. No session files are present here.
- [ ] Approve exact paper run IDs in `results/paper_manifest.csv`.
- [ ] Export and independently compare curated aggregates with the paper.
- [ ] Run `python scripts/validate_release.py --strict-release`.
- [ ] Run the pre-init and post-commit secret/personal-path scan.
- [ ] Inspect the standalone Git tree and remote before any push.

