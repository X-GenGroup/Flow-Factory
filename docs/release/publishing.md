# Publishing a reviewed release

Release preparation and publication are separate. Review and merge the complete stack first;
then select the integration commit whose code, package version and accepted evidence agree.
Do not move an existing tag. v0.3.0 (`17e8be7c`) and v0.4.0 (`fbb038e3`) already exist.

## Before merge

The package workflow builds the sdist and wheel, validates runtime contents, installs both outside
the checkout and records environment and hashes. For a CUDA image preview, dispatch the existing
container workflow from the reviewed branch with its full 40-character source SHA:

```bash
gh workflow run container.yml --ref <reviewed-branch> \
  -f source_sha=<full-commit-sha> -f publish=false
```

Leave `tag` empty. The preview cannot publish any image aliases. It retains the source/recipe
identities, resolved environment, image metadata and offline CPU smoke. Local GPU lifecycle
receipts remain separate and identify their installed package and selected test commands.

## After merge

1. Verify the integration main checks and evidence continuity; inspect any changed merge tree.
   Compare runtime source hashes with the tested artifacts. A changed runtime owner needs its own
   focused evidence before release.
2. Confirm `pyproject.toml` says `0.5.0`, then create an annotated `v0.5.0` tag on that reviewed
   integration commit and push it. Tag creation is an explicit maintainer delivery action.
3. Dispatch the stable workflow **from the same tag**:

   ```bash
   gh workflow run release.yml --ref v0.5.0 -f tag=v0.5.0
   ```

4. The identity job checks the tag commit, checked-out workflow, clean source and package version.
   Packages are rebuilt and installed; the CUDA image is built, checked offline and published to
   GHCR only after registry access succeeds. The publishing job attaches wheel, sdist, checksums,
   release notes and an evidence archive to a draft Release, then makes it public.
5. Inspect the public Release assets and GHCR digest. Record the final tag/commit, image digest,
   workflow URL and checksums in the release records. Do not claim publication based only on a
   merged PR or a successful preview.

The workflow does not publish to PyPI. The image aliases `v0.5.0` and `0.5.0` are immutable;
`latest` advances only for a stable version. Source and recipe must match for new releases.
Historical v0.1–v0.4 container builds retain their recorded separate-recipe behavior.

## Failure and evidence handling

No test failure is converted into publication success. A failed package/image job prevents the
GitHub Release publishing job. If asset upload fails after creating a draft, keep it as a draft,
inspect the failed run and finish uploading that run's verified assets before publishing. If the
image job already succeeded, rerun only failed jobs instead of overwriting versioned image tags.
Never overwrite a public Release's assets to conceal a changed source or failed check.

Actions package and container artifacts last 90 days. The stable workflow copies its own build,
installation and image records into the Release evidence archive with SHA256 checksums. Historical
numerical receipts keep their original source identities and remain indexed by the shipped
roadmap handoffs; this archive does not relabel or rerun them. Preserve downloaded original GPU
receipts separately when their source artifacts are nearing expiry.
