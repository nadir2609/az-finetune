<#
Starts (or resumes) the Azerbaijani StyleTTS2 fine-tune.
  .\scripts\train.ps1                 # fresh run, 7.5 h hard cap
  .\scripts\train.ps1 -Resume         # continue from newest checkpoint in the config's log_dir
  .\scripts\train.ps1 -MaxHours 3     # total wall-clock cap across the run and any auto-resumes
  .\scripts\train.ps1 -Force          # skip the free-VRAM precheck (warn instead of stop)
  .\scripts\train.ps1 -DryRun         # print the command(s) that would run; never starts training
On a crash (non-zero exit) the script auto-resumes from the newest checkpoint, up to -MaxRestarts
times, as long as time remains in the -MaxHours wall-clock budget (measured from this launcher's
own start, across all restarts - each restart is handed only the remaining budget, so the total
never exceeds -MaxHours).
Monitor in a second terminal:  .\.venv\Scripts\python.exe scripts\monitor.py --follow 60
TensorBoard:                    .\.venv\Scripts\tensorboard.exe --logdir <log_dir>\tensorboard
The run directory is log_dir in configs\config_ft_az.yml (set by make_config.py --log_dir).
#>
param(
    [double]$MaxHours = 7.5,
    [switch]$Resume,
    [switch]$Force,
    [int]$MaxRestarts = 3,
    [switch]$DryRun
)

function Get-NewestCheckpoint {
    param([string]$Dir)
    Get-ChildItem $Dir -Filter "epoch_2nd_*.pth" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Config = Join-Path $Root "configs\config_ft_az.yml"
$Calibration = Join-Path $Root "configs\calibration.json"
if (-not (Test-Path $Config)) { throw "Missing $Config - run scripts\make_config.py first." }
# train_finetune.py writes checkpoints to the config's log_dir, so resume must look there too
$logLine = Select-String -Path $Config -Pattern '^log_dir:\s*(.+)$' | Select-Object -First 1
if (-not $logLine) { throw "No log_dir in $Config" }
$LogDir = $logLine.Matches[0].Groups[1].Value.Trim().Trim("'").Trim('"')
if ((-not $Resume) -and (Get-NewestCheckpoint -Dir $LogDir)) {
    throw "$LogDir already holds checkpoints; a fresh run there would auto-resume from them after a crash. " + `
          "If this config was built for that same run, continue it with -Resume. For a new run, rebuild the " + `
          "config with make_config.py --pretrained <start weights> --log_dir <new empty dir>."
}
$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# --- Free-VRAM precheck (replaces a simple "used > 1500 MiB" warning) ---
$requiredFreeGB = 6.866
if (Test-Path $Calibration) {
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
        $ckpt = Get-NewestCheckpoint -Dir $LogDir
        if (-not $ckpt) { throw "No checkpoint in $LogDir to resume from." }
        Write-Host "Resuming from $($ckpt.FullName)"
        $argsList += @("--resume_from", $ckpt.FullName)
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
    $ckpt = Get-NewestCheckpoint -Dir $LogDir

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

Write-Host "Synthesize: .\.venv\Scripts\python.exe scripts\synthesize.py --checkpoint <$LogDir\epoch_2nd_XXXXX.pth>"
