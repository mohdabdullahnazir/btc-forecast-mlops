# MLOps Assignment 3

Build and demonstrate a seven-day BTC-USD closing-price forecast. The walkthrough groups the work by the assignment rubric and provides PowerShell commands for Docker Desktop on Windows.

## Before you start

Open the extracted project folder in VS Code. Open **Terminal → New Terminal**, select PowerShell 7, and confirm you can see `docker-compose.yml` in the current directory.

Install Python 3.12, Git, Docker Desktop with the Linux engine, Minikube, kubectl, and Helm. The Docker images and GitHub Actions use Python 3.12. If your system Python is 3.13, use a separate 3.12 virtual environment for the commands below. Start Docker Desktop before running Docker or Minikube commands. You also need GitHub and Docker Hub accounts for Part B.

```powershell
py -3.12 --version
git --version
docker version
docker compose version
minikube version
kubectl version --client
helm version
```

Plan for roughly 4 CPU cores and 6 GB available to Minikube for the full demo. On a smaller machine, complete Parts A–B first, stop the Compose stack while running standalone Kubeflow, then restart Compose for Part E. Create an evidence directory for outputs.

```powershell
New-Item -ItemType Directory -Force models,evidence | Out-Null
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements-dev.txt
```

`$py` is a PowerShell variable. Set it again in any new terminal where you need it. All paths below are relative to the repository root.

## 1. Train the model locally

```powershell
& $py -m src.etl_fetch --output models/raw.csv
& $py -m src.preprocess --input models/raw.csv --output models/features.csv
& $py -m src.train_forecast --input models/features.csv --output models/model.json --registry models/versions
Get-Content models/model.json
```

The first command downloads daily BTC-USD OHLCV candles from Yahoo Finance, beginning in 2015 and excluding the incomplete current UTC day. Preprocessing sorts by date, removes invalid rows and duplicates, and uses the latest uninterrupted daily segment. It builds seven lagged log returns, the log distance from the 7- and 30-day moving averages, and seven-day return volatility. The target is the next day's log return. Open, high, low, and volume are stored for reproducibility. The model uses close-derived predictors so it can recompute them from its own forecasts.

The trainer fits a standardized Ridge autoregression with alpha 10. It recursively predicts the last seven observed closes from a single forecast origin. Neither the scaler nor the evaluation fit sees the held-out observations. The naive baseline repeats the last pre-holdout close. The gate requires MAPE ≤ 25% and RMSE ≤ the greater of 1.5 times the baseline RMSE and 0.5% of the last pre-holdout close. A passing candidate is refit on all known data and saved as a versioned JSON model. `/metrics` reports the pre-refit holdout scores. MAE and RMSE are in USD. MAPE is a percentage. A single holdout provides limited evidence, so report whether the candidate beat the baseline and what that comparison can support.

If the market-data provider returns no rows, check the network and retry. If the gate rejects a candidate, capture its metrics, keep the active model, and run again when new candles are available. Changing the model or gate requires documenting the change and evaluating on an independent window.

## 2. Part A · Run Flask and test the API

```powershell
& $py -m pytest -q
& $py -m black --check app src dags kubeflow tests load scripts
& $py -m flake8 app src dags kubeflow tests load scripts
$env:MODEL_ROOT = (Resolve-Path models).Path
& $py -m flask --app app.app run --port 5000
```

In another PowerShell window, run the requests below. Stop the development server with **Ctrl+C** before using Compose. The Docker container uses Gunicorn to provide the production WSGI entry point on Linux.

```powershell
$api = "http://127.0.0.1:5000"
Invoke-RestMethod "$api/health" | ConvertTo-Json
Invoke-RestMethod "$api/metrics" | ConvertTo-Json
Invoke-RestMethod -Method Post "$api/predict" -ContentType "application/json" -Body '{}' | ConvertTo-Json -Depth 6
$custom = @{ forecast_days = 3; last_date = '2025-12-31'; historical_closes = @(80000) * 30 } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post "$api/predict" -ContentType "application/json" -Body $custom | ConvertTo-Json -Depth 6
try {
  Invoke-RestMethod -Method Post "$api/predict" -ContentType "application/json" -Body '{"forecast_days":8}'
} catch {
  $_.Exception.Response.StatusCode
  $_.ErrorDetails.Message
}
```

| Request | Expected result |
| --- | --- |
| GET /health | 200 with status, worker uptime, stage, and model version. A missing model returns 503. |
| POST /predict with `{}` | 200 with seven dated forecasts after the latest observed UTC date. |
| POST /predict with custom data | 200 with one to seven forecasts. Supply 30–3650 numeric closes in oldest-first daily order and the date of the last close. |
| POST /predict with malformed or invalid input | 400 and a JSON error. |
| GET /metrics | 200 with the holdout dates, MAE, RMSE, MAPE and baseline RMSE. |
| Unexpected server failure | 500 and a JSON error. Details go to server logs. |

Capture the test summary, a valid seven-day JSON response, the metrics response and one 400 example. The repository's test data is synthetic and used only for tests.

## 3. Parts B and C1 · Run Docker Compose and Airflow

Create `.env` once. Preserve it when restarting Compose because the PostgreSQL volume uses the original password. It is excluded from Git.

```powershell
if (-not (Test-Path .env)) {
  $settings = foreach ($key in 'POSTGRES_PASSWORD','AIRFLOW_ADMIN_PASSWORD','AIRFLOW_WEB_SECRET','RELOAD_TOKEN') {
    $value = & $py -c "import secrets; print(secrets.token_urlsafe(32))"
    "$key=$value"
  }
  $settings + 'K8S_MODEL_RELOAD_URL=' | Set-Content -Path .env -Encoding ascii
}
docker compose config --quiet
docker compose --progress plain build 2>&1 | Tee-Object evidence/docker-build.txt
docker compose up -d
docker compose ps -a
docker compose logs --tail 50 airflow-init
```

The first start initializes PostgreSQL, the Airflow metadata schema and a shared model volume. `airflow-init` and `storage-init` exit successfully after setup. Airflow uses LocalExecutor, so the scheduler launches task processes and PostgreSQL stores orchestration metadata. The initial API returns 503 until a model is registered.

Open [Airflow](http://127.0.0.1:8080), sign in as `admin` with the `AIRFLOW_ADMIN_PASSWORD` value from your local `.env`, find `btc_etl_pipeline`, and unpause it. Click **Trigger DAG**, select the manual run, and open **Graph**. Alternatively, use the CLI below.

```powershell
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose exec airflow-scheduler airflow dags unpause btc_etl_pipeline
docker compose exec airflow-scheduler airflow dags trigger btc_etl_pipeline
docker compose exec airflow-scheduler airflow dags list-runs -d btc_etl_pipeline
Invoke-RestMethod http://127.0.0.1:5000/health
Invoke-RestMethod http://127.0.0.1:5000/metrics
```

The daily DAG runs at 01:00 UTC, has one active run at a time, and retries failed tasks. The four boxes in the graph should be `fetch_data`, `preprocess_features`, `train_evaluate_register`, and `reload_api`. Run files and versions persist on the shared volume. The final task installs the active artifact atomically and calls the API reload endpoint with a token. Capture the graph after all tasks succeed and the last two task logs showing metrics and the served version. For a failure, click the failed task and open **Logs** before retrying.

### Push GitHub Actions and registry images

1. In Docker Hub, create public repositories named **btc-api** and **btc-trainer**. Create a Docker Hub personal access token with push access.
2. Create an empty GitHub repository called `btc-forecast-mlops`. In its **Settings → Secrets and variables → Actions**, add **DOCKER_USERNAME** and **DOCKER_PASSWORD**. Use your Docker Hub token in `DOCKER_PASSWORD`.
3. In the project root, run the following commands. Substitute your GitHub username or organization.

```powershell
git init -b main
git add .
git status --short
git commit -m "Implement BTC forecasting MLOps pipeline"
git remote add origin https://github.com/YOUR_GITHUB_NAME/btc-forecast-mlops.git
git push -u origin main
```

The GitHub Actions workflow runs Black, Flake8, and pytest on main pushes and pull requests. It then builds the API and trainer images. Only main pushes publish both images to Docker Hub, tagged with the full Git commit SHA and `latest`. The `kubeflow-spec` job compiles a YAML artifact linked to that commit's trainer image. Verify that **quality**, **images**, and **kubeflow-spec** all succeed before taking a screenshot. Copy your repository URL and both public Docker Hub URLs for submission. Record the commit SHA with `git rev-parse HEAD`.

## 4. Part D · Deploy the Kubernetes API

Start a **single-node Minikube** cluster with Docker Desktop running. The PVC dynamically provisions a PV using Minikube's default StorageClass. All replicas share it on the same node. This verifies pod recovery on one node. Describe the node-level limitation in the report.

```powershell
minikube start --driver=docker --cpus=4 --memory=6144
kubectl config use-context minikube
minikube addons enable metrics-server
kubectl get nodes
kubectl get storageclass
docker build -f app/Dockerfile -t btc-api:dev .
docker build -f src/Dockerfile -t btc-trainer:dev .
minikube image load btc-api:dev
minikube image load btc-trainer:dev
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-secret.yaml -f k8s/pvc.yaml
```

Use the same reload token for Airflow and Kubernetes. The manifest creates an empty Secret. This command supplies the value from your local environment file without putting the token in Git or your command history.

```powershell
$reloadToken = ((Get-Content .env | Where-Object { $_ -like 'RELOAD_TOKEN=*' }) -replace '^RELOAD_TOKEN=', '')
if (-not $reloadToken) { throw 'Set RELOAD_TOKEN in .env first.' }
$secretPatch = @{ stringData = @{ RELOAD_TOKEN = $reloadToken } } | ConvertTo-Json -Compress
$patchPath = Join-Path $env:TEMP ("btc-secret-" + [guid]::NewGuid() + ".json")
try {
  $secretPatch | Set-Content -Path $patchPath -Encoding ascii
  kubectl -n btc-forecast patch secret btc-api-secrets --type merge --patch-file $patchPath
  if ($LASTEXITCODE -ne 0) { throw 'Could not set the Kubernetes Secret.' }
} finally {
  Remove-Item -LiteralPath $patchPath -ErrorAction SilentlyContinue
  Remove-Variable reloadToken,secretPatch -ErrorAction SilentlyContinue
}
```

Seed the PVC before starting the pods. The Job runs the same ingestion and training modules and exits after it has written the initial artifact. Wait for three API pods before applying the HPA, which can later scale down to two when idle.

```powershell
kubectl apply -f k8s/seed-job.yaml
kubectl -n btc-forecast wait --for=condition=complete job/btc-model-seed --timeout=300s
kubectl -n btc-forecast logs job/btc-model-seed -c trainer
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl -n btc-forecast rollout status deployment/btc-api --timeout=180s
kubectl -n btc-forecast get deployment,pods,svc,pvc -o wide | Tee-Object evidence/kubernetes-initial.txt
kubectl get pv
kubectl apply -f k8s/hpa.yaml
kubectl -n btc-forecast top pods
```

If you want to demonstrate the actual registry image after publishing, set the Deployment to its immutable tag. Substitute the Docker Hub account and full SHA. For the seed Job to use a registry image on a fresh cluster, edit the `image` field in `k8s/seed-job.yaml` before applying it.

```powershell
kubectl -n btc-forecast set image deployment/btc-api api=docker.io/YOUR_DOCKERHUB_NAME/btc-api:YOUR_COMMIT_SHA
kubectl -n btc-forecast rollout status deployment/btc-api --timeout=180s
```

### Enable ingress and test the Kubernetes service

The Windows walkthrough installs F5 NGINX Ingress Controller through Helm. Its local service remains a ClusterIP. A Windows port-forward reaches the controller and the controller distributes traffic across the API replicas.

```powershell
helm upgrade --install btc-nginx oci://ghcr.io/nginx/charts/nginx-ingress --version 2.9.1 --namespace ingress-nginx --create-namespace --set controller.service.type=ClusterIP --set controller.enableCustomResources=false --wait --timeout 5m
kubectl apply -f k8s/ingress.yaml
kubectl get ingressclass
kubectl -n btc-forecast get pods,svc,hpa,ingress,pvc -o wide | Tee-Object evidence/kubernetes-resources.txt
kubectl -n ingress-nginx get svc
```

In a new terminal, start this port-forward and leave it running. If `$ingressService` is empty, substitute the controller Service name printed above.

```powershell
$ingressService = kubectl -n ingress-nginx get svc -l app.kubernetes.io/instance=btc-nginx -o jsonpath='{.items[0].metadata.name}'
kubectl -n ingress-nginx port-forward "svc/$ingressService" 8081:80
```

In the working terminal, send requests with the configured host header. To demonstrate the separate NodePort Service, run `minikube service btc-api-nodeport -n btc-forecast --url` in another terminal and append `/health` to the URL printed there.

```powershell
$cluster = "http://127.0.0.1:8081"
$hostHeader = @{ Host = "btc-forecast.local" }
Invoke-RestMethod -Headers $hostHeader "$cluster/health"
Invoke-RestMethod -Headers $hostHeader "$cluster/metrics"
Invoke-RestMethod -Method Post -Headers $hostHeader "$cluster/predict" -ContentType "application/json" -Body '{}' | ConvertTo-Json -Depth 6 | Tee-Object evidence/cluster-forecast.json
```

### Load test and capture autoscaling

Use the ingress URL for the load test so the requests can reach every API replica. Wait until `kubectl -n btc-forecast top pods` returns CPU values. Start the HPA watch in one terminal and sample the HPA in another.

```powershell
kubectl -n btc-forecast get hpa --watch
```

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/sample_hpa.ps1 -Samples 60
```

In a third terminal, install the load-test tools and send traffic to your local ingress.

```powershell
py -3.12 -m venv .load-venv
$loadPy = ".\.load-venv\Scripts\python.exe"
& $loadPy -m pip install locust matplotlib
& $loadPy -m locust -f load/locustfile.py --headless -u 150 -r 20 -t 3m --host http://127.0.0.1:8081 --csv evidence/locust --csv-full-history
& $loadPy scripts/chart_scaling.py evidence/hpa.csv evidence/hpa-scaling.png
```

Record the starting, peak, and ending replica counts, CPU utilization, requests per second, p95 latency and failure rate. If CPU does not cross 60%, check the Locust failures and the metrics pipeline, then increase the client count or duration on your own cluster. Once load stops, the default HPA behavior can take several minutes to scale down. Capture the actual values you observed.

## 5. Part C2 · Compile and run Kubeflow Pipelines

Use the trainer image published in Part B. Replace `YOUR_DOCKERHUB_NAME` and `YOUR_COMMIT_SHA` with real values. Use a separate Python environment for the KFP SDK.

```powershell
py -3.12 -m venv .kfp-venv
$kfpPy = ".\.kfp-venv\Scripts\python.exe"
& $kfpPy -m pip install -r requirements-kfp.txt
$env:BTC_TRAINER_IMAGE = "docker.io/YOUR_DOCKERHUB_NAME/btc-trainer:YOUR_COMMIT_SHA"
& $kfpPy kubeflow/btc_kfp_pipeline.py --output kubeflow/btc_pipeline.yaml
Select-String -Path kubeflow/btc_pipeline.yaml -Pattern 'image:'
```

The included YAML is reference IR for `btc-trainer:dev`. Replace it with SDK compiler output before submission. You can also use the compiled YAML from the successful GitHub Actions **btc-kubeflow-pipeline** artifact.

For a live run, install the standalone development backend on Minikube. Confirm the current context first. Standalone KFP provides a UI, artifact storage and execution metadata.

```powershell
kubectl config current-context
$kfpVersion = "2.18.0"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$kfpVersion"
kubectl wait --for=condition=established --timeout=120s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$kfpVersion"
kubectl -n kubeflow get pods
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8082:80
```

Open [Kubeflow Pipelines](http://127.0.0.1:8082) and upload the compiled YAML under **Pipelines**. Create a run from that pipeline. The graph should show **fetch-btc → engineer-features → train-7day**. Open **Run output** or the training step to see the model artifact, and check the step logs for metrics. The pipeline disables caching so scheduled runs fetch fresh candles. For the recurring demo, create a recurring run and choose a one-day interval. Airflow manages promotion to Flask. Kubeflow trains independently and stores its datasets and model in its artifact store. Capture the Kubernetes API and HPA evidence before running Kubeflow if your laptop needs those resources back.

## 6. Part E · Verify model updates and pod recovery

### Verify Airflow updates the live Kubernetes API

Keep your ingress port-forward running. Start a separate port-forward for the private reload route.

```powershell
kubectl -n btc-forecast port-forward svc/btc-api 5002:80
```

Replace the `K8S_MODEL_RELOAD_URL=` line in `.env` with the following line, then recreate the scheduler so it sees the new environment. Execute the `docker compose exec` command to confirm the scheduler can reach the bridge.

```dotenv
K8S_MODEL_RELOAD_URL=http://host.docker.internal:5002/internal/reload
```

```powershell
docker compose up -d --force-recreate airflow-scheduler
docker compose exec airflow-scheduler python -c "from urllib.request import urlopen; print(urlopen('http://host.docker.internal:5002/health', timeout=10).read().decode())"
```

If Docker Desktop cannot reach the loopback listener, restart the bridge on `--address 0.0.0.0` for the demo on your own development machine, then repeat the connectivity check. Stop that port-forward when the demo is complete. If the selected pod is replaced, restart this port-forward before triggering Airflow again.

In your working terminal, record the old model version, trigger a manual DAG run, and wait until all its tasks are green. The final task should log both endpoints confirming the same model version. The cluster stores versioned JSON and the corresponding OHLCV CSV on the PVC. All replicas read the active artifact from that shared storage.

```powershell
$cluster = "http://127.0.0.1:8081"
$hostHeader = @{ Host = "btc-forecast.local" }
$before = Invoke-RestMethod -Headers $hostHeader "$cluster/health"
$before | ConvertTo-Json | Set-Content evidence/model-before.json
docker compose exec airflow-scheduler airflow dags trigger btc_etl_pipeline
# Wait for the exact manual run to succeed in the Airflow UI.
$after = Invoke-RestMethod -Headers $hostHeader "$cluster/health"
$after | ConvertTo-Json | Set-Content evidence/model-after.json
if ($before.model_version -eq $after.model_version) { throw "The Kubernetes API still serves the old version." }
$prediction = Invoke-RestMethod -Method Post -Headers $hostHeader "$cluster/predict" -ContentType "application/json" -Body '{}'
if ($prediction.model_version -ne $after.model_version -or $prediction.predictions.Count -ne 7) { throw "Model or forecast check failed." }
$prediction | ConvertTo-Json -Depth 6 | Tee-Object evidence/cluster-forecast.json
```

A same-day retrain may create a new version with identical prices because its input candles have not changed. In the report, record both the model version and the most recent training date. Continuous desktop-to-cluster updates depend on the demo port-forward and a running Docker scheduler.

### Demonstrate self-healing

Run this on your assignment cluster only. Choose a current pod from the deployment, delete it, and capture a new pod reaching Ready. Open the `get pods --watch` command in another terminal before deletion if you want to record the transition.

```powershell
kubectl -n btc-forecast get pods -l app=btc-api -o wide
$victim = kubectl -n btc-forecast get pods -l app=btc-api -o jsonpath='{.items[0].metadata.name}'
kubectl -n btc-forecast delete pod $victim
kubectl -n btc-forecast rollout status deployment/btc-api --timeout=180s
kubectl -n btc-forecast get pods -l app=btc-api -o wide | Tee-Object evidence/kubernetes-after-deletion.txt
Invoke-RestMethod -Headers $hostHeader "$cluster/health"
```

Record the old and new pod names, the readiness time you observed, and the model version after replacement. Run deployment verification before stopping any port-forwards.

## 7. Assemble the 4–6 page PDF and submit

The `report` folder includes a five-page template. Open `Report_Template.html` in a browser, replace the prompts, and paste cropped screenshots into the outlined evidence boxes. Use **Print to PDF** and select **Save as PDF**. Turn off browser headers and footers. Check the resulting PDF for legibility and a total of 4–6 pages.

| Page | Evidence and explanation |
| --- | --- |
| 1 · Architecture and forecast | Group details, GitHub and Docker Hub links, flow diagram, source dates, feature and holdout design, and validation gate. |
| 2 · Flask and CI/CD | Seven-day prediction, invalid request, `/metrics`, test summary, Docker build log and GitHub Actions screenshot. |
| 3 · Orchestration | Successful manual Airflow graph, metrics and reload logs, compiled KFP YAML and KFP graph or model artifact. |
| 4 · Kubernetes and load | `kubectl get pods,svc,hpa,ingress,pvc`, probes and resource settings, Locust chart or output, HPA chart and measured findings. |
| 5 · Resilience and results | Deleted and replacement pod names, before/after model versions, final Kubernetes forecast and limits observed in your demo. |

Submit the GitHub repository URL, public Docker Hub repository URLs, and the completed PDF. The repository should contain the source code, Dockerfiles, Compose file, DAG, compiled Kubeflow YAML, Kubernetes manifests and CI workflow. Keep passwords and tokens out of screenshots and commits.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `docker` unavailable | Start Docker Desktop, select Linux containers and reopen the terminal. |
| Airflow UI missing | Run `docker compose ps -a` and `docker compose logs --tail 100 airflow-init airflow-webserver`. A different app may occupy port 8080. |
| DAG task fails during fetch | Read its log. Yahoo access may be blocked or rate-limited. Retry after confirming that connectivity works inside the relevant container. |
| Seed job is stuck or failed | Inspect `kubectl -n btc-forecast describe job btc-model-seed` and its trainer logs. After correcting the issue, delete the failed Job and reapply its manifest to retry. |
| API pod is pending or fails probes | Inspect `kubectl -n btc-forecast describe pod POD_NAME`, verify the PVC is Bound, and read the seed logs. |
| HPA CPU is `<unknown>` | Wait for resource samples, check the metrics-server pod in `kube-system`, and verify `kubectl -n btc-forecast top pods`. |
| Ingress returns 404 | Verify the Host header, ingress class, controller logs and `kubectl -n btc-forecast describe ingress btc-api`. |
| Kubernetes reload returns 401 | Check that the Compose token and cluster Secret match. Restart the API Deployment after changing its Secret because the token is read at process startup. |
| KFP trainer gets `ImagePullBackOff` | Match the trainer name and commit SHA to the public Docker Hub image and regenerate the YAML. |
| GitHub workflow fails at Docker push | Check Docker Hub repositories and GitHub Actions secrets. PR builds skip publishing. |

## Official references

- [Airflow 2.10.5 with Docker Compose](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/docker-compose/index.html)
- [Kubeflow container components and compilation](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/container-components/)
- [Kubeflow Pipelines standalone installation](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)
- [GitHub Actions Docker publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)
- [Minikube Docker driver on Windows](https://minikube.sigs.k8s.io/docs/drivers/docker/)
- [Kubernetes Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)
- [NGINX Ingress Helm installation](https://docs.nginx.com/nginx-ingress-controller/install/helm/open-source/)
- [yfinance download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
