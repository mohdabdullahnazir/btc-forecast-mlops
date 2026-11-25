param(
    [string]$OutputPath = "evidence/hpa.csv",
    [int]$Samples = 60
)

$parent = Split-Path -Parent $OutputPath
if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
"timestamp,cpu_pct,replicas,desired" | Set-Content -Path $OutputPath -Encoding ascii
for ($i = 0; $i -lt $Samples; $i++) {
    $hpa = kubectl -n btc-forecast get hpa btc-api -o json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Could not query the BTC HPA." }
    $cpu = $null
    if ($hpa.status.currentMetrics) {
        $cpu = $hpa.status.currentMetrics[0].resource.current.averageUtilization
    }
    $stamp = (Get-Date).ToUniversalTime().ToString("o")
    "$stamp,$cpu,$($hpa.status.currentReplicas),$($hpa.status.desiredReplicas)" |
        Add-Content -Path $OutputPath -Encoding ascii
    Start-Sleep -Seconds 5
}
