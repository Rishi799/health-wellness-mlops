# Render deployment and rollback runbook

Owner: Rishab (Docker/deployment), with Rishi (GitHub CI/CD). This is an educational API demo, not a clinically validated risk service. The model card records calibration and subgroup limitations.

## 1. Publish a verified image

Have a teammate review and approve changes before merging to `main` under the team workflow.

1. On GitHub, check that the PR's **Offline tests (no secrets)**, **Data checks and tests**, and **Docker image builds and starts** jobs pass. A draft PR runs only the offline job; mark it ready for review to run the full jobs.
2. Merge after approval, then open Actions. Wait for **CI** on `main` and its resulting **CD** run to succeed.
3. Copy the full 40-character main commit SHA from GitHub. Keep this as the release's application identity. CD publishes `ghcr.io/rishi799/health-wellness-api:<SHA>` and `:latest`.
4. Before Render is configured, CD explicitly reports **image publication only; no service was deployed**. That is expected for the first release.

CD only accepts successful CI for a same-repository push or manual CI run on `main`. To retry publication later, rerun CI on `main`; there is no unchecked manual CD shortcut. A stale commit is rejected if main has advanced.

## 2. Create the Render service manually

1. Open [Render Dashboard](https://dashboard.render.com/). Sign in or create your own account with GitHub. Complete any authentication or account terms yourself. Use the Free compute plan; do not select a paid plan or add paid resources for this demo.
2. Choose **New > Web Service > Existing Image**.
3. Enter `ghcr.io/rishi799/health-wellness-api:latest`, then **Connect**. The image is public, so leave registry credentials empty. This is an image deployment: do not enter the GitHub repository as the source.
4. Configure the service:

| Setting | Value |
| --- | --- |
| Name | `health-wellness-api` (Render may append a suffix if the name is taken) |
| Region | Singapore |
| Instance / compute plan | Free |
| Environment variable | `PORT` = `8000` |
| Advanced: Health Check Path | `/health` |
| Docker command override | Leave empty; use the image's command |

5. Do not add DagsHub credentials, an MLflow server, a database, a disk, or a GitHub token. The image already contains the model and metadata.
6. Select **Create/Deploy Web Service**. Watch Events/Logs until the deployment is Live. Timing varies; image pull, model startup and a Free instance waking can take longer than the estimate in the team message.
7. Copy the **actual HTTPS URL shown by Render**. Do not assume the requested name was available. The root `/` may return 404 because this is an API; use `/health` or `/docs`.

Render supports public GHCR images and requires Linux amd64, which the workflow builds. See [prebuilt image deployment](https://render.com/docs/deploying-an-image) and [web service configuration](https://render.com/docs/web-services).

## 3. Verify the public deployment

First open these paths in your browser:

| Path | Expected result |
| --- | --- |
| `/health` | HTTP 200, `status: healthy`, `model_loaded: true`, model version `2` |
| `/model-info` | `model_name: cardiometabolic-risk-model`, 21 features, `model_version: 2` |
| `/model-info` | `service_git_commit` equals the full SHA published in step 1 |
| `/docs` | Swagger UI with `POST /predict` |

`git_commit` in model metadata describes the training run. **`service_git_commit` describes the application image. They should not be assumed equal.**

On `/docs`, expand **POST /predict > Try it out**, replace the request with this synthetic example, then choose **Execute**:

```json
{"age":55,"sex":1,"bmi":30.0,"sbp":140.0,"told_high_bp":1.0}
```

Expect HTTP 200 with `probability`, `risk_score`, `risk_band`, `model_version` and the educational disclaimer. Try `education: 2.5` in the same request and confirm HTTP 422. Do not submit personal health information during deployment checks.

For the complete automated check, run this in PowerShell from the repository folder. Replace both placeholders with your actual URL and the full published main SHA:

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test --base-url https://YOUR-ACTUAL-SERVICE.onrender.com --expected-commit YOUR_FULL_MAIN_SHA --expected-model-version 2 --attempts 60 --delay 10 --timeout 20
```

A PASS verifies health, model and application identity, a prediction, score/band consistency, docs, and rejection of the invalid category. A healthy old image cannot pass the expected-commit check. The checker sends synthetic inputs only and needs no deployment secret.

## 4. Enable GitHub-to-Render delivery

Image-backed Render services do **not** redeploy just because `latest` changed in GHCR. The deploy hook connects image publication to a new Render deployment. See [Render deploy hooks](https://render.com/docs/deploy-hooks).

1. In Render, open the service's **Settings > Deploy Hook** and copy the hook privately.
2. Open the [repository Actions settings](https://github.com/Rishi799/health-wellness-mlops/settings/secrets/actions). Under **Secrets**, create/update repository secret **`RENDER_DEPLOY_HOOK`** with that value. The workflow uses this exact name, not `RENDER_DEPLOY_HOOK_URL` from Render's generic example.
3. Under **Variables**, create **`RENDER_SERVICE_URL`** containing only the public origin, e.g. `https://YOUR-ACTUAL-SERVICE.onrender.com`. Include no `/docs`, `/health`, query string, password, or hook in this value.
4. If you lack repository settings access, have Rishi do steps 2-3. Send the hook only to Rishi privately, as requested by the team. Do not paste it into this chat, a group, an issue, a PR, a screenshot, a committed file, or the public variable.
5. Keep existing `DAGSHUB_USER` and `DAGSHUB_TOKEN` secrets in GitHub for CI's artifact download. They are not needed in Render.
6. Run **Actions > CI > Run workflow > main**, or let the next reviewed merge trigger it. After CI succeeds, CD builds and tests the image before pushing it, asks Render to deploy that exact SHA via `imgURL`, and polls the public API until `service_git_commit` matches. Render accepting the hook alone is not counted as deployment success.

If the hook is configured but `RENDER_SERVICE_URL` is missing, CD fails before sending a deploy request. If the hook was exposed, regenerate it in Render and replace the GitHub secret privately.

Optional private message to Rishi (replace placeholders privately):

```text
The Render service is Live: <PUBLIC_URL>
Model version: 2. Application commit: <FULL_MAIN_SHA>.
Please set repository secret RENDER_DEPLOY_HOOK to the hook I am sharing privately, and repository variable RENDER_SERVICE_URL to the public URL above. Then rerun CI on main and confirm CD's public verification passes.
```

Inviting Rishi to the Render workspace is optional. Use your own workspace membership UI and confirm the intended email and any plan implications before sending an invitation; it is not required to deploy this image.

## 5. Rehearse image rollback

1. Record the currently verified service URL, application SHA, model version, and successful deploy in Render.
2. Keep the previous successful image SHA/digest in GHCR. Do not delete images required for rollback.
3. In Render's **Deploys** history, select a previous successful deploy and use **Rollback** if available. Alternatively edit the service's image reference to `ghcr.io/rishi799/health-wellness-api:<PREVIOUS_SUCCESSFUL_SHA>` and use **Manual Deploy > Deploy latest reference**.
4. Wait for Live and run the smoke test with the **previous application SHA**. Confirm `/model-info` reports the expected old service commit. If rolling back to the historical image published before these fixes, it lacks `service_git_commit`; use that only as an explicitly limited compatibility check, not as a successful new release verification. Prefer two successful hardened releases for the full drill.
5. Restore the chosen current SHA/reference and verify again. After a temporary explicit SHA override in Settings, restore `:latest` if that is the intended default; the CD hook still overrides each automated deployment with its exact published SHA.
6. Record previous/current application identities, model versions, times, smoke results, and reason. Moving the MLflow champion alias alone does not change a model baked into an existing image.

Render re-pulls the referenced image for rollback. Check that both the target digest and restore digest remain available, and check that the recorded tag still resolves to the intended digest. [Image rollback requirements](https://render.com/docs/deploying-an-image#rollback-support)

### Recorded Phase 2 drill

On 10 November 2026, the public API passed a baseline smoke check on `0acf8cf`. Both SHA tags still resolved to the digests recorded by Render. The two deploys below used Render's rollback action and passed the full public smoke test after becoming live.

| Step | Application commit | Model | Live at (IST) | Render deploy | Smoke |
| --- | --- | --- | --- | --- | --- |
| Roll back | `9704e4370b48e172244103ad220aae5ac0a22e9d` | `2` | 00:23:33 | `dep-dbp1d12jnfac7394hon0` | PASS |
| Restore | `0acf8cf2f92584bb58f8bf258142be238a949524` | `2` | 00:24:42 | `dep-dbp1di3l550s73c1no10` | PASS |

- Rollback image digest `sha256:79a9ba1dbf4b17b7ecfa5cf7c6ac934a7a883dd9ff3d96ba8137a04c7a243219`
- Restored image digest `sha256:8e469efe370e04bce52f752fde0f47e43e59a0ce7e8f70706e632f55d9ed5790`

The service's default image reference is `:latest` and Render checks `/health`. GitHub CD supplies a SHA tag through the deploy hook. Model version 2 comes from MLflow run `077b778198d448628bf29fbf8c4fd9e6`. [Public model information](https://health-wellness-api.onrender.com/model-info) · [Release CI](https://github.com/Rishi799/health-wellness-mlops/actions/runs/42282663898) · [Release CD](https://github.com/Rishi799/health-wellness-mlops/actions/runs/42283291228)

## 6. Troubleshooting and evidence

| Symptom | Check |
| --- | --- |
| Image not found / unauthorized | Verify the full `ghcr.io/rishi799/health-wellness-api` path and that the selected SHA tag exists publicly |
| Port detection failure | Set `PORT=8000`, leave Docker command override empty; the image binds `0.0.0.0` and honors PORT |
| `/health` returns 503 | Inspect private Render logs for artifact/schema/startup errors; the public API deliberately omits internal paths |
| Smoke reports old application | Wait for the new deploy; verify the image SHA; a hook acknowledgement does not mean the new instance is serving |
| Health works but invalid categories pass | Old image is running; publish and deploy the hardening change |
| Service pauses while idle | Free services spin down after 15 minutes without inbound traffic and must wake on the next request |
| Hook step fails | Verify the GitHub secret privately and service URL variable; inspect Render Events; do not print the hook |

Free services have ephemeral storage and usage limits. Do not rely on local prediction logs surviving a restart; Phase 3 persistence needs a separate design. See [Free service limitations](https://render.com/docs/free). Keep the demo on Free unless you deliberately choose otherwise.

For each release, retain the public URL, application SHA and image digest, model version and run ID, passing CI/CD links, smoke results, and any rollback evidence. The `phase2-review` tag identifies the release verified in the drill above. [Phase 2 tag](https://github.com/Rishi799/health-wellness-mlops/tree/phase2-review) · [Model card](model_card.md)
