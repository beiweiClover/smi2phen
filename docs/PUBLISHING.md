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
- complete resource distribution: GitHub Release asset
  `smi2phen-resources-v0.1.0.tar.gz`;
- the project owner confirmed that the selected 18-file resource snapshot may be redistributed.

The MIT license covers the smi2phen source code only. Preserve the upstream source, version,
license, and attribution notes recorded in `resources/manifest.json`.

## 1. Local preflight

From the project root:

```bash
python -m pytest
python -m compileall -q src scripts tests
python -m ruff check .
python scripts/build_resource_bundle.py --resource-root /path/to/audited/resources
python scripts/download_resources.py --dry-run
docker compose config
git status --short
```

## 2. Build and test the complete resource asset

Build the deterministic full archive:

```powershell
$sourceResources = Read-Host "Absolute path to the audited resource root"
python scripts/build_resource_bundle.py `
  --resource-root $sourceResources `
  --version v0.1.0
```

Expected upload files are created below the ignored `.release-staging/` directory:

```text
smi2phen-resources-v0.1.0.tar.gz
SHA256SUMS
resource-bundle-metadata.json
```

Before uploading, confirm the archive metadata equals `resources/manifest.json` and test the exact
archive through the public downloader code:

```powershell
$bundle = (Resolve-Path ".release-staging\smi2phen-resources-v0.1.0.tar.gz").Path
$bundleUrl = ([Uri]$bundle).AbsoluteUri
python scripts/download_resources.py `
  --resource-root ".release-staging\download-test" `
  --bundle-url $bundleUrl
python scripts/check_resources.py `
  --resource-root ".release-staging\download-test" `
  --mode enhanced
```

Do not upload the unused `kg-base/kg.csv`, run artifacts, databases, caches, or the source
`.docker-data` directory itself.

## 3. Publish the GitHub Release resource asset

1. Push the reviewed source commit to `main`.
2. Open the repository's **Releases** page and choose **Draft a new release**.
3. Create/select tag `v0.1.0` at the reviewed source commit.
4. Set the title to `smi2phen v0.1.0`.
5. Paste the prepared release notes from `docs/RELEASE_NOTES_v0.1.0.md`.
6. Upload all three files from `.release-staging/`.
7. Publish the release.
8. In a logged-out browser, confirm the archive is downloadable from:

   ```text
   https://github.com/beiweiClover/smi2phen/releases/download/v0.1.0/smi2phen-resources-v0.1.0.tar.gz
   ```

9. In a new empty directory, run `python scripts/download_resources.py` and
   `python scripts/check_resources.py --mode enhanced`. Record publication as successful only if
   all 18 files verify.

## 4. Configure the release identity

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

## 5. Push source

The GitHub initialization commit contains the MIT `LICENSE` and has been merged without overwriting
either history. Confirm the configured remote, then push:

```bash
git remote -v
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` and use `git remote set-url origin ...`
only after confirming the exact target. Never place a token in a remote URL.

The release tag is created through the GitHub Release workflow above. Record:

```bash
git rev-parse HEAD
git ls-remote origin refs/tags/v0.1.0
```

## 6. Build and push the GHCR image

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

## 7. Clean-room verification

Use a new empty directory and a new Compose project name. Do not copy local images, resources,
caches, or volumes into it.

```bash
git clone https://github.com/beiweiClover/smi2phen.git smi2phen-clean
cd smi2phen-clean
git checkout v0.1.0
cp .env.example .env
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
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
