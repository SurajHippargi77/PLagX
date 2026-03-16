# PLagX - Cross-Language Plagiarism Intelligence
# Usage:
#   .\run.ps1 download            Download Wikipedia reference documents
#   .\run.ps1 status              Show downloaded reference documents
#   .\run.ps1 check -f file.txt   Check a text file for plagiarism
#   .\run.ps1 check -f file.pdf   Check a PDF for plagiarism
#   .\run.ps1 check -t "text..."  Check inline text

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Create venv if it doesn't exist
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate venv
Write-Host "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Install dependencies if needed
Write-Host "Checking dependencies..."
pip install -q -r requirements.txt

# Pass all arguments to plagx CLI
Write-Host "Running PLagX..."
python -m plagx @args
