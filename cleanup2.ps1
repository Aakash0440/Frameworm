# cleanup.ps1 — Frameworm repo tidying script
# Run from the root of your project:  .\cleanup.ps1
# Uses 'git mv' if inside a git repo, plain Move-Item otherwise.

$ROOT = Get-Location

# Detect git
$useGit = $false
try {
    git -C $ROOT rev-parse --git-dir 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $useGit = $true
        Write-Host "Git repo detected - using git mv" -ForegroundColor Green
    }
} catch {}

if (-not $useGit) {
    Write-Host "Not a git repo - using Move-Item" -ForegroundColor Yellow
}

function Move-Entry($src, $dest) {
    if (Test-Path $src) {
        if ($useGit) {
            git mv $src $dest
        } else {
            Move-Item -Path $src -Destination $dest
        }
        Write-Host "  moved: $src -> $dest" -ForegroundColor Cyan
    } else {
        Write-Host "  skip (not found): $src" -ForegroundColor DarkGray
    }
}

function Remove-Entry($src) {
    if (Test-Path $src) {
        Remove-Item -Path $src -Force
        Write-Host "  deleted: $src" -ForegroundColor Red
    } else {
        Write-Host "  skip (not found): $src" -ForegroundColor DarkGray
    }
}

# Step 1: Move design/plan docs to docs\design\
Write-Host ""
Write-Host "Step 1: Move *_design.md / *_plan.md -> docs\design\"

New-Item -ItemType Directory -Force -Path "docs\design" | Out-Null

$designDocs = @(
    "advanced_training_design.md","bayesian_optimization_design.md",
    "callbacks_design.md","cli_design.md","concurrency_patterns.md",
    "dag_design.md","day21_design.md","day22_design.md","day23_design.md",
    "dcgan_design.md","ddpm_design.md","deployment_design.md",
    "distributed_optimization_design.md","distributed_training_design.md",
    "docs_design.md","error_patterns.md","error_system_design.md",
    "exception_hierarchy.md","execution_design.md","experiment_tracking_design.md",
    "graph_algorithms.md","hyperparameter_search_design.md","logging_design.md",
    "metrics_design.md","parallel_execution_design.md","performance_plan.md",
    "plugin_discovery_design.md","plugin_system_design.md","registry_patterns.md",
    "training_patterns.md","training_system_design.md","type_system_plan.md",
    "web_ui_design.md","test_plan_registry.md","DAYS11-12_COMPLETE_SUMMARY.md",
    "PROJECT_PROGRESS.md"
)

foreach ($f in $designDocs) { Move-Entry $f "docs\design\$f" }

# Step 2: Move useful test scripts to tests\
Write-Host ""
Write-Host "Step 2: Move useful test scripts -> tests\"

$keepTests = @(
    "test_config_script.py","test_discover.py",
    "test_validation_script.py","test_final_verification.py","test_env.py"
)

foreach ($f in $keepTests) { Move-Entry $f "tests\$f" }

# Step 3: Delete throwaway debug/scratch files
Write-Host ""
Write-Host "Step 3: Delete throwaway files"

$deleteFiles = @(
    "debug_discovery.py","debug_tes_plugins.py","custom_plugin_fixed.py",
    "test_server.py","test_cli.sh","test_env.yaml"
)

foreach ($f in $deleteFiles) { Remove-Entry $f }

# Step 4: Remove duplicate latest.pt from root
Write-Host ""
Write-Host "Step 4: Remove duplicate latest.pt from root"
Remove-Entry "latest.pt"

# Step 5: Move posts\ -> docs\design\ then remove folder
Write-Host ""
Write-Host "Step 5: Move posts\ content -> docs\design\"
if (Test-Path "posts") {
    Get-ChildItem "posts" | ForEach-Object {
        Move-Entry "posts\$($_.Name)" "docs\design\$($_.Name)"
    }
    Remove-Item "posts" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "  removed dir: posts\" -ForegroundColor Red
}

# Step 6: Move experiment_schema.sql -> experiments\
Write-Host ""
Write-Host "Step 6: Move experiment_schema.sql -> experiments\"
Move-Entry "experiment_schema.sql" "experiments\experiment_schema.sql"

Write-Host ""
Write-Host "Done! Run 'git status' to review all moves before committing." -ForegroundColor Green