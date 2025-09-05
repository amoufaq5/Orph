# scripts/run_orph.ps1
param(
  [string]$OutDir = ".\data\raw",
  [string]$PubMedTerm = "randomized controlled trial[pt] OR review[pt]",
  [string]$CTExpr = "COVID-19",
  [string]$MinDate,
  [string]$MaxDate
)

$ErrorActionPreference = "Stop"

chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at .\.venv. Activate or create it first." }

if (-not (Test-Path ".\src")) { throw "Run this script from the project root (where .\src exists)." }

if (-not $env:SCRAPER_EMAIL) { $env:SCRAPER_EMAIL = "you@yourdomain.com" }

# Build PubMed args dynamically
$pubmedArgs = @("--out", $OutDir, "--term", $PubMedTerm)
if ($MinDate) { $pubmedArgs += @("--mindate", $MinDate) }
if ($MaxDate) { $pubmedArgs += @("--maxdate", $MaxDate) }

Write-Host "[run] PubMed ESummary scrape → $OutDir"
& $py -m src.data_prep.scrapers.pubmed @pubmedArgs

Write-Host "[run] ClinicalTrials.gov scrape → $OutDir"
& $py -m src.data_prep.scrapers.clinicaltrials --out $OutDir --expr $CTExpr

Write-Host "[run] Done."
