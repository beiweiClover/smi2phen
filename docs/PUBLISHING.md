# Publishing runbook

This runbook separates local release preparation from actions that require the repository owner's
accounts. The release target is `https://github.com/beiweiClover/smi2phen`, and the fixed container
target is `ghcr.io/beiweiclover/smi2phen:v0.1.0`. Do not record a successful upload, digest, or
clean-room result until the corresponding command has completed.

## Decisions required before a public release

The following decisions are recorded:

- public GitHub repository: `https://github.com/beiweiClover/smi2phen`;
- project source license: MIT;
- lower-case GHCR namespace: `ghcr.io/beiweiclover`;
- the verified GPS files continue to use pinned official URLs unless a tested Google Drive mirror
  is supplied.

The MIT license covers the smi2phen source code only. It does not override third-party resource
licenses or grant permission to redistribute manual-only resources.

## 1. Local preflight

From the project root:

```bash
python -m pytest
python -m compileall -q src scripts tests
python -m ruff check .
python scripts/download_resources.py --dry-run
docker compose config
git status --short
```

Review `resources/manifest.json`. Do not upload any resource whose
`redistribution_status` is `manual_only_needs_review`.

## 2. Optional Google Drive mirror for GPS

The runtime can already retrieve the nine GPS files from pinned, byte-verified official GitHub
URLs. A Google Drive mirror is optional.

If a mirror is required:

1. Run `python scripts/download_resources.py`. It downloads the nine approved GPS files and reports
   the other resources as manual.
2. Upload only the GPS paths classified under
   `release_audit.third_party_upstream_downloadable`. Preserve their relative folder structure.
3. Include the Apache-2.0 license and attribution to `Bin-Chen-Lab/GPS`, commit
   `c11668aaa08a68ec3e2e9d93d79ca4dd1956ba98`.
4. Set each Drive file to “Anyone with the link” and viewer-only.
5. Record the Drive file ID, revision, and sharing link. For script downloads, use only a raw
   download URL returned by Drive (for example, the file's `webContentLink`) that works without an
   account. Do not assume that a manually constructed URL is a stable public API. Google documents
   browser downloads through `webContentLink`; its `files.get?alt=media` API flow normally uses an
   authorization token:
   <https://developers.google.com/workspace/drive/api/guides/manage-downloads>.
6. Replace that resource's `download_url` only after the anonymous URL returns the resource bytes,
   not an HTML login/permission page, in a logged-out browser or clean shell. If no anonymous raw
   URL is available, keep the pinned official GPS URL and document Drive as a manual mirror.
7. Download into a new empty resource root and verify every file:

   ```bash
   python scripts/download_resources.py --resource-root .release-staging/drive-test
   python scripts/check_resources.py --resource-root .release-staging/drive-test
   ```

The full checker will remain non-ready until the manual-only Core resources are supplied. A Drive
HTML permission page will fail SHA-256 verification; do not treat it as a resource file.

For each mirrored file, retain this record:

```text
resource_id | filename | size | sha256 | public URL | Drive revision | expected_relative_path
```

Do not upload NCBI, NetInfer, PPI, or transformed KG files from the audited local snapshot until
their exact provenance and redistribution rights are resolved.

## 3. Configure the release identity

Set the fixed image in `.env.example` and the Compose fallback to:

```text
ghcr.io/beiweiclover/smi2phen:v0.1.0
```

Replace repository placeholders in `README.md` with:

```text
https://github.com/beiweiClover/smi2phen.git
```

The default README command must use `v0.1.0`, not `latest`. `latest` may be maintained as a
convenience alias.

## 4. Push source and create the release tag

The GitHub initialization commit contains the MIT `LICENSE` and has been merged without overwriting
either history. Confirm the configured remote, then push:

```bash
git remote -v
git push -u origin main
```

Create the tag only after the GHCR push and clean-room verification have passed:

```bash
git tag -a v0.1.0 -m "smi2phen v0.1.0"
git push origin v0.1.0
```

If `origin` already exists, inspect it with `git remote -v` and use `git remote set-url origin ...`
only after confirming the exact target. Never place a token in a remote URL.

Record:

```bash
git rev-parse HEAD
git rev-parse v0.1.0
```

## 5. Build and push the GHCR image

Build from the tagged source tree. Do not add `.local-resources`, `.runtime`, `.env`, or staging
files to the Docker context.

```bash
docker build -f docker/Dockerfile.unified \
  -t ghcr.io/beiweiclover/smi2phen:v0.1.0 .
docker image inspect ghcr.io/beiweiclover/smi2phen:v0.1.0
docker login ghcr.io -u beiweiClover
docker push ghcr.io/beiweiclover/smi2phen:v0.1.0
docker tag ghcr.io/beiweiclover/smi2phen:v0.1.0 \
  ghcr.io/beiweiclover/smi2phen:latest
docker push ghcr.io/beiweiclover/smi2phen:latest
docker buildx imagetools inspect ghcr.io/beiweiclover/smi2phen:v0.1.0
```

Enter the package token only at Docker's password prompt. Do not paste it into a file, command
argument, README, shell transcript, or Git remote. The token needs the minimum package permission
required by the owner's GHCR policy.

Make the GHCR package public if anonymous `docker compose pull` is part of the release requirement.
Record the repository-qualified digest returned by the registry inspection.

## 6. Clean-room verification

Use a new empty directory and a new Compose project name. Do not copy local images, resources,
caches, or volumes into it.

```bash
git clone https://github.com/beiweiClover/smi2phen.git smi2phen-clean
cd smi2phen-clean
git checkout v0.1.0
cp .env.example .env
python scripts/download_resources.py
python scripts/check_resources.py
docker compose -p smi2phen-clean pull
docker compose -p smi2phen-clean up -d
docker compose -p smi2phen-clean ps
curl http://127.0.0.1:8000/healthz
python scripts/validate_unified.py --provided-targets
```

Confirm that:

- the image was pulled from GHCR rather than satisfied by an existing local tag;
- Redis, API, and Worker are healthy;
- the Web page responds at `http://127.0.0.1:8000/`;
- the Worker actually claims and completes the validation task;
- the clean state database has no historical user sessions or tasks.

After recording results:

```bash
docker compose -p smi2phen-clean down -v
```

Failure to obtain every required scientific resource is a release blocker for a full scientific
run, not permission to report the run as passed.
