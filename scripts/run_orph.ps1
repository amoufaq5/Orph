# scripts/run_orph.ps1
param(
  [string]$OutDir = ".\data\raw",
  [string]$PubMedTerm = "randomized controlled trial[pt] OR review[pt]",
  [string]$CTExpr = "COVID-19",
  [string]$MinDate = $null,
  [string]$MaxDate = $null
)

$ErrorActionPreference = "Stop"

# Ensure venv
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Error "Python venv not found at .\.venv. Activate or create it first."
}
$py = ".\.venv\Scripts\python.exe"

# Polite contact; set once in your session or here
if (-not $env:SCRAPER_EMAIL) { $env:SCRAPER_EMAIL = "you@yourdomain.com" }

# (Optional) NCBI key for higher rate limits
# if (-not $env:NCBI_API_KEY) { $env:NCBI_API_KEY = "<5fa3b3391d1cb5dd412e9092373d68385c08>" }

Write-Host "[run] PubMed ESummary scrape → $OutDir"
& $py "src\data_prep\scrapers\pubmed.py" --out $OutDir --term $PubMedTerm --mindate $MinDate --maxdate $MaxDate

Write-Host "[run] ClinicalTrials.gov scrape → $OutDir"
& $py "src\data_prep\scrapers\clinicaltrials.py" --out $OutDir --expr $CTExpr

Write-Host "[run] Done."
