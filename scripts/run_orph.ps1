# scripts/run_orph.ps1
param(
  [string]$OutDir = ".\data\raw",
  [string]$PubMedTerm = "randomized controlled trial[pt] OR review[pt]",
  [string]$CTExpr = "COVID-19",
  [string]$MinDate,
  [string]$MaxDate,
  [int]$PubMedChunk = 500,
  [int]$CTPageSize = 100,
  [string]$OpenFDAQuery = $null,  # e.g. 'openfda.generic_name:"ibuprofen"'
  [int]$OpenFDAMax = 50000,
  [int]$DailyMedPageSize = 100,
  [int]$DailyMedMaxPages = 500
)

$ErrorActionPreference = "Stop"

# UTF-8 arrows and logs
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

# venv python
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at .\.venv. Activate or create it first." }

# Ensure running from repo root
if (-not (Test-Path ".\src")) { throw "Run this script from the project root (where .\src exists)." }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

# Required contact header for NCBI
if (-not $env:SCRAPER_EMAIL) { throw "Set SCRAPER_EMAIL (e.g., `$env:SCRAPER_EMAIL='you@domain.com'`)." }

# --- PubMed ---
$pubmedArgs = @("--out", $OutDir, "--term", $PubMedTerm, "--chunk", $PubMedChunk)
if ($MinDate) { $pubmedArgs += @("--mindate", $MinDate) }
if ($MaxDate) { $pubmedArgs += @("--maxdate", $MaxDate) }

Write-Host "[run] PubMed ESummary scrape → $OutDir"
& $py -m src.data_prep.scrapers.pubmed @pubmedArgs

# --- ClinicalTrials.gov ---
Write-Host "[run] ClinicalTrials.gov scrape → $OutDir"
& $py -m src.data_prep.scrapers.clinicaltrials --out $OutDir --expr $CTExpr --page_size $CTPageSize

# --- OpenFDA (drug labels) ---
$openFDAArgs = @("--out", $OutDir, "--max_records", $OpenFDAMax)
if ($OpenFDAQuery) { $openFDAArgs += @("--query", $OpenFDAQuery) }

Write-Host "[run] OpenFDA Labels scrape → $OutDir"
& $py -m src.data_prep.scrapers.openfda_labels @openFDAArgs

# --- DailyMed SPLs ---
Write-Host "[run] DailyMed SPLs scrape → $OutDir"
& $py -m src.data_prep.scrapers.dailymed --out $OutDir --page_size $DailyMedPageSize --max_pages $DailyMedMaxPages

Write-Host "[run] Done."
