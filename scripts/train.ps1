<#
Starts (or resumes) the Azerbaijani StyleTTS2 fine-tune.
  .\scripts\train.ps1                 # fresh run, 7.5 h hard cap
  .\scripts\train.ps1 -Resume         # continue from newest checkpoint in runs\az_ft
  .\scripts\train.ps1 -MaxHours 3     # custom cap (use remaining budget when resuming)
  .\scripts\train.ps1 -Force          # skip the free-VRAM precheck (warn instead of stop)
  .\scripts\train.ps1 -DryRun         # print the command(s) that would run; never starts training
On a crash (non-zero exit) the script auto-resumes from the newest checkpoint, up to -MaxRestarts
times, as long as time remains in the -MaxHours wall-clock budget (measured from this launcher's
own start, across all restarts).
Monitor in a second terminal:  .\.venv\Scripts\python.exe scripts\monitor.py --follow 60
TensorBoard:                    .\.venv\Scripts\tensorboard.exe --logdir runs\az_ft\tensorboard
#>
param(
    [double]$MaxHours = 7.5,
    [switch]$Resume,
    [switch]$Force,
    [int]$MaxRestarts = 3,
    [switch]$DryRun,
    # Internal test hook: overrides the calibrated required-free-VRAM value so the precheck's
    # failure path can be exercised without training and without needing another app to actually
    # hog VRAM. Not documented above; leave 0 in normal use.
    [double]$TestRequiredFreeGB = 0,
    # Internal test hook: comma-separated exit codes (one per attempt, last one repeats) used to
    # replace the real training invocation with `python -c "import sys; sys.exit(N)"` so the
    # crash/auto-resume/restart-budget loop can be exercised without ever calling train_finetune.py.
    # Not documented above; leave empty in normal use.
    [string]$TestStubExitCodes = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Config = Join-Path $Root "configs\config_ft_az.yml"
$LogDir = Join-Path $Root "runs\az_ft"
$Calibration = Join-Path $Root "configs\calibration.json"
if (-not (Test-Path $Config)) { throw "Missing $Config - run scripts\make_config.py first." }
$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# --- Free-VRAM precheck (replaces a simple "used > 1500 MiB" warning) ---
$requiredFreeGB = 6.866
if ($TestRequiredFreeGB -gt 0) {
    $requiredFreeGB = $TestRequiredFreeGB
} elseif (Test-Path $Calibration) {
    $cal = Get-Content $Calibration -Raw | ConvertFrom-Json
    if ($cal.free_gb_at_launch) { $requiredFreeGB = [double]$cal.free_gb_at_launch }
} else {
    Write-Warning "Missing $Calibration - using default required free VRAM of $requiredFreeGB GB."
}
$freeMiB = [double](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
$freeGB = $freeMiB / 1024
$thresholdGB = $requiredFreeGB - 0.2
if ($freeGB -lt $thresholdGB) {
    $msg = "Only $([math]::Round($freeGB,2)) GB VRAM free; need >= $([math]::Round($thresholdGB,2)) GB " + `
           "(calibrated launch requirement $requiredFreeGB GB). Close browsers/games/Steam/other GPU apps."
    if ($Force) {
        Write-Warning $msg
    } else {
        throw "$msg  (rerun with -Force to override)"
    }
}

New-Item -ItemType Directory -Force $LogDir | Out-Null
Set-Location (Join-Path $Root "StyleTTS2")

# --- Bounded auto-resume loop: total wall-clock across restarts never exceeds $MaxHours ---
$launcherStart = Get-Date
$restarts = 0
$resumeThisLaunch = $Resume.IsPresent

while ($true) {
    $elapsedH = ((Get-Date) - $launcherStart).TotalHours
    $remainingH = $MaxHours - $elapsedH
    if ($remainingH -le 0) {
        throw "No time budget remaining ($([math]::Round($remainingH,2)) h) before starting - stopping."
    }

    $argsList = @("train_finetune.py", "-p", $Config, "--max_hours", $remainingH)
    if ($resumeThisLaunch) {
        $ckpt = Get-ChildItem $LogDir -Filter "epoch_2nd_*.pth" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $ckpt) { throw "No checkpoint in $LogDir to resume from." }
        Write-Host "Resuming from $($ckpt.FullName)"
        $argsList += @("--resume_from", $ckpt.FullName)
    }

    if ($TestStubExitCodes) {
        $codes = $TestStubExitCodes -split ","
        $idx = [Math]::Min($restarts, $codes.Length - 1)
        $argsList = @("-c", "import sys; sys.exit($($codes[$idx].Trim()))")
    }

    $startMsg = "Training started $(Get-Date) - attempt $($restarts + 1) (restart $restarts of $MaxRestarts) - " + `
                "budget $([math]::Round($remainingH,2)) h. Log: $LogDir\train.log"
    Write-Host $startMsg

    if ($DryRun) {
        Write-Host "[DryRun] $Py $($argsList -join ' ')"
        break
    }

    # PS 5.1 wraps native stderr lines in ErrorRecords; with "Stop" the first warning would kill the run.
    $ErrorActionPreference = "Continue"
    & $Py @argsList 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "console.log") -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"

    if ($exitCode -eq 0) {
        Write-Host "Finished $(Get-Date) with exit code 0."
        break
    }

    Write-Warning "Training process exited with code $exitCode."
    $elapsedH = ((Get-Date) - $launcherStart).TotalHours
    $remainingH = $MaxHours - $elapsedH
    $ckpt = Get-ChildItem $LogDir -Filter "epoch_2nd_*.pth" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if (($restarts -lt $MaxRestarts) -and ($remainingH -gt 0.25) -and $ckpt) {
        $restarts++
        Write-Host "crashed (exit $exitCode), resuming from $($ckpt.FullName) with $([math]::Round($remainingH,2)) h left"
        $resumeThisLaunch = $true
        continue
    } else {
        $stopMsg = "Training stopped: exit code $exitCode, restarts used $restarts/$MaxRestarts, " + `
                   "remaining budget $([math]::Round($remainingH,2)) h, checkpoint found: $([bool]$ckpt)."
        throw $stopMsg
    }
}

Write-Host "Synthesize: .\.venv\Scripts\python.exe scripts\synthesize.py --checkpoint <runs\az_ft\epoch_2nd_XXXXX.pth>"
