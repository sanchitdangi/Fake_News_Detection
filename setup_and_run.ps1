# setup_and_run.ps1
$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

Write-Host "=========================================="
Write-Host " Upgraded Fake News Detection ML Setup"
Write-Host "=========================================="

$PythonExe = "python"
$PipExe = "pip"
$CondaExe = ""

$pythonWorks = $false
try {
    $ver = & $PythonExe --version 2>&1
    if ($ver -match "Python") {
        $pythonWorks = $true
    }
} catch {
}

if ($pythonWorks) {
    Write-Host "[+] Python is installed globally and working."
} else {
    Write-Host "[-] Python not found. Installing local Miniconda..."
    $MinicondaDir = Join-Path $ProjectDir "miniconda"
    $PythonExe = Join-Path $MinicondaDir "python.exe"
    $PipExe = Join-Path $MinicondaDir "Scripts\pip.exe"
    $CondaExe = Join-Path $MinicondaDir "Scripts\conda.exe"
    
    if (-not (Test-Path $PythonExe)) {
        Write-Host "[!] Installing portable Miniconda locally..."
        $InstallerPath = Join-Path $ProjectDir "Miniconda3-latest-Windows-x86_64.exe"
        if (-not (Test-Path $InstallerPath)) {
            Write-Host "    Downloading Miniconda installer..."
            Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $InstallerPath
        }
        Write-Host "    Running silent installation (this may take a few minutes)..."
        $InstallArgs = "/InstallationType=JustMe /RegisterPython=0 /S /D=$MinicondaDir"
        Start-Process -FilePath $InstallerPath -ArgumentList $InstallArgs -Wait -NoNewWindow
        Write-Host "[+] Miniconda installed locally."
    } else {
        Write-Host "[+] Local Miniconda found."
    }
}

Write-Host "`n[1/5] Verifying & Installing Required Packages..."
# Install packages
& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Pip package installation failed." }

# Fix MKL-conflict DLL entries by force-installing nomkl
if ($CondaExe -ne "" -and (Test-Path $CondaExe)) {
    Write-Host "    Applying nomkl configuration to prevent threadpoolctl/BLAS segfaults on Windows..."
    & $CondaExe install -y --override-channels -c conda-forge nomkl
    if ($LASTEXITCODE -ne 0) { throw "Conda nomkl installation failed." }
}

Write-Host "`n[2/5] Downloading & Combining Datasets (Kaggle + LIAR)..."
& $PythonExe download_dataset.py
if ($LASTEXITCODE -ne 0) { throw "Dataset download/combination failed." }

Write-Host "`n[3/5] Running Model Training & Deep Learning Fine-Tuning..."
& $PythonExe run_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Pipeline execution failed." }

Write-Host "`n[4/5] Generating Visual Plots and Confusion Matrices..."
& $PythonExe draw_plots.py
if ($LASTEXITCODE -ne 0) { throw "Plot drawing failed." }

Write-Host "`n[5/5] Compiling DOCX, PDF, and PPTX Presentation Slides..."
& $PythonExe generate_reports.py
if ($LASTEXITCODE -ne 0) { throw "Report compilation failed." }

Write-Host "`n=========================================="
Write-Host " ALL STEPS COMPLETED SUCCESSFULLY!"
Write-Host "=========================================="
Write-Host "Final Submission Checklist:"
Write-Host "[x] Kaggle + Politifact LIAR Combined Dataset"
Write-Host "[x] Code reproducibility set (fixed seeds)"
Write-Host "[x] Traditional baselines & DistilBERT model trained"
Write-Host "[x] saved_models.pkl and distilbert_fake_news/ weights serialized"
Write-Host "[x] LIME explanations (Real/Fake HTML) generated"
Write-Host "[x] Accuracy comparisons, ROC curves, and confusion matrix PNGs generated"
Write-Host "[x] IEEE_Report.docx generated in report/"
Write-Host "[x] IEEE_Report.pdf generated in report/"
Write-Host "[x] Presentation.pptx generated in report/"
Write-Host "[x] Streamlit app.py interface created"
Write-Host "=========================================="
Write-Host "To launch the interactive app, run: ./miniconda/Scripts/streamlit.exe run app.py"
Write-Host "=========================================="
