# Frozen Publication Release & DOI Archival

This repository backs a dissertation. Before a submission or defense, freeze the exact code and text that produced the reported numbers and archive that snapshot with a persistent identifier (DOI), so a reviewer can retrieve the precise state the claims rest on. A moving `main` branch cannot serve that role.

This file is the checklist. Two steps (tagging, and minting the DOI on Zenodo) require the repository owner's account and are marked **[owner action]**; they cannot be done from inside an automated session.

---

## Pre-freeze verification (do this first)

1. **Confirm the README, `docs/REPRODUCTION_MAP.md`, and `results_manifest.json` agree** on every headline number and its tier (confirmatory / exploratory / archived).
2. **Run the full verification suite** and confirm 453/453 pass:
   ```
   python chapter4Experiments/verify_chapter4.py
   python chapter5Experiments/verify_chapter5.py
   python chapter5Experiments/verify_experiment_zero.py
   python chapter5Experiments/verify_reproduce_chapter5.py
   python chapter5Experiments/verify_baselines.py
   python chapter6Experiments/verify_chapter6.py
   python chapter6Experiments/verify_reproduce_chapter6.py
   python chapter7Experiments/verify_chapter7.py
   python chapter7Experiments/verify_extract_utilities.py
   python experiments/ch5_4class/verify_ch5_4class.py
   python experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py
   python experiments/ablation/verify_ablation.py
   python validation/verify_validators.py
   python experiments/confirmatory/verify_confirmatory.py
   ```
   Remember: a green run verifies **code behavior, not scientific validity** (see `docs/VERIFICATION_METHODOLOGY.md`). It is a precondition for freezing, not a warrant for the claims.
3. **Regenerate the confirmatory numbers from authorized data (if available).** The confirmatory pipeline (SRP-64 + participant-grouped `StratifiedGroupKFold` + fold-local preprocessing) that produces the headline 60.6% / 53.2% numbers is committed at `experiments/confirmatory/run_confirmatory_validation.py`; `results_manifest.json` now points every confirmatory row at it. If you hold the authorized SHAPE data, run it and confirm agreement before freezing:
   ```
   python experiments/confirmatory/run_confirmatory_validation.py \
       --data3 ./batch_data --data4 ./categories --out confirmatory_results.json
   python experiments/confirmatory/check_against_manifest.py confirmatory_results.json --tol 1.5
   ```
   The SHAPE data is restricted and not shipped, so if you are freezing without it, state in the release notes that the confirmatory numbers were last regenerated on the authorized data and that CI verifies the pipeline machinery (not the numbers) via `verify_confirmatory.py`.

## Bind the artifacts with a release manifest

On the exact commit you are about to tag, generate the integrity manifest that
binds the dissertation source, PDF, results manifest, and full figure set to
that commit, and commit it:

```
python scripts/make_release_manifest.py --pdf ARSPI-Net_Defense.pdf --out RELEASE_MANIFEST.json
git add RELEASE_MANIFEST.json && git commit -m "chore: release manifest for vX.Y.Z"
```

Re-running on the same commit reproduces byte-identical checksums, so anyone
can verify the archived state. (This checks integrity, not scientific
validity.)

## Tag the frozen release **[owner action]**

Choose a semantic version (e.g. `v1.0.0` for the submitted/defended version) and tag the exact commit:

```
git tag -a v1.0.0 -m "Frozen publication release: ARSPI-Net dissertation submission"
git push origin v1.0.0
```

Then draft a GitHub Release from that tag. Suggested release notes:

- One-line summary: corrected confirmatory results (SRP-64, 5×5-fold participant-grouped, fold-local).
- Link to `results_manifest.json` and `docs/REPRODUCTION_MAP.md`.
- The verification statement: 453/453 tests pass; tests verify code behavior, not scientific validity.
- Known gap: confirmatory regeneration script pending (if not yet committed).

## Mint a DOI on Zenodo **[owner action]**

1. Sign in to [Zenodo](https://zenodo.org) with the GitHub account and enable the **GitHub → Zenodo** integration for `TheAwesomeAndy/dissoAdventureExperiments` (Zenodo account → *GitHub* → toggle the repo **on**). Zenodo archives a new snapshot automatically **each time a GitHub Release is published** for an enabled repo.
2. Publish the GitHub Release created above. Zenodo ingests the tarball and mints a version DOI plus a **concept DOI** that always resolves to the latest version.
3. Copy the DOI back into the repository:
   - Set `doi:` and the `identifiers` DOI in [`CITATION.cff`](CITATION.cff) (replace the `10.5281/zenodo.XXXXXXX` placeholder), and set `version` and `date-released` to match the tag.
   - Add the DOI badge to the top of `README.md`.
   - Commit as `docs: record Zenodo DOI for vX.Y.Z frozen release`.

   > Because the DOI only exists after the release is published, this is a small follow-up commit **after** the frozen tag. That is expected and standard; the tag remains the archived scientific state, and this commit only records the identifier that points at it.

## After archival

- The DOI is the citable, immutable reference for reviewers. Cite it alongside the dissertation.
- Continue development on `main`; the tag and its DOI stay pinned to the frozen state.
- For a revised submission, repeat this checklist with a new version tag; Zenodo's concept DOI will track the series.
