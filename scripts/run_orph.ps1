# scripts/run_orph.ps1
param(
  [string]$OutDir = ".\data\raw",
  [string]$PubMedTerm = "randomized controlled trial[pt] OR review[pt]",
  [string]$CTExpr = "COVID-19",
  [string]$MinDate = $null,
  [string]$MaxDate = $null
)

$ErrorActionPreference = "Stop"

# --- UTF-8 console (fixes the weird â†’ arrow) ---
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

# --- venv python ---
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at .\.venv. Activate or create it first." }

# --- Required layout: run from repo root where src/ exists ---
if (-not (Test-Path ".\src")) { throw "Run this script from the project root (where .\src exists)." }

# --- polite contact header for scrapers (recommended by NCBI) ---
if (-not $env:SCRAPER_EMAIL) { $env:SCRAPER_EMAIL = "you@yourdomain.com" }
# Optional: NCBI API key to raise rate limits
# if (-not $env:NCBI_API_KEY) { $env:NCBI_API_KEY = "<your-key>" }

Write-Host "[run] PubMed ESummary scrape → $OutDir"
& $py -m src.data_prep.scrapers.pubmed --out $OutDir --term $PubMedTerm --mindate $MinDate --maxdate $MaxDate

Write-Host "[run] ClinicalTrials.gov scrape → $OutDir"
& $py -m src.data_prep.scrapers.clinicaltrials --out $OutDir --expr $CTExpr

Write-Host "[run] Done."
