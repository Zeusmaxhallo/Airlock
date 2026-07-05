# Airlock Corpus Compatibility Test (v2)
# - prüft nur echte Contract-Crates (cdylib-Target) statt --workspace
#   -> umgeht osmosis-test-tube (Go/cgo), libfuzzer (MSVC), alloy (MSRV 1.86)
# - repariert bekannte Nightly-inkompatible Lockfile-Pins (ahash stdsimd,
#   proc-macro2 proc_macro_span_shrink) per `cargo update --precise`
#
# Aufruf:  powershell -ExecutionPolicy Bypass -File .\run_corpus.ps1
# Optional: -Repo <name>  (nur ein Repo/Unit testen)

param([string]$Repo = "")

$ErrorActionPreference = "Continue"
$root       = $PSScriptRoot
$corpus     = Join-Path $root "corpus"
$resultsDir = Join-Path $root "corpus-results"
# wie rust-toolchain.toml / TOOLCHAIN in main.rs
$toolchain  = "nightly-2026-07-04"

New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

# 1. Airlock bauen
Write-Host "== Baue airlock =="
Push-Location (Join-Path $root "airlock")
rustup run $toolchain cargo build
if ($LASTEXITCODE -ne 0) { Write-Error "airlock build fehlgeschlagen"; Pop-Location; exit 1 }
$airlock = (Resolve-Path ".\target\debug\airlock.exe").Path
Pop-Location

# Bekannte Crates, die per Nightly-Detection entfernte Features aktivieren
function Repair-Lockfile([string]$dir) {
    $lock = Join-Path $dir "Cargo.lock"
    if (-not (Test-Path $lock)) { return @() }
    $txt   = Get-Content $lock -Raw
    $fixes = @()
    foreach ($m in [regex]::Matches($txt, 'name = "(ahash|proc-macro2|alloy)"\r?\nversion = "([0-9.]+)"')) {
        $n = $m.Groups[1].Value
        $v = [version]$m.Groups[2].Value
        $targets = @()
        if     ($n -eq "ahash" -and $v.Minor -eq 7 -and $v -lt [version]"0.7.8")  { $targets = @("0.7.8")  }
        elseif ($n -eq "ahash" -and $v.Minor -eq 8 -and $v -lt [version]"0.8.7")  { $targets = @("0.8.11") }
        elseif ($n -eq "proc-macro2" -and $v -lt [version]"1.0.66")               { $targets = @("1.0.86") }
        # alloy >=1.0.23 verlangt rustc 1.86; 1.0.22 = letzte mit MSRV 1.85, 1.0.9 = letzte mit 1.82
        elseif ($n -eq "alloy" -and $v -ge [version]"1.0.23")                     { $targets = @("1.0.22", "1.0.9") }
        foreach ($target in $targets) {
            rustup run $toolchain cargo update "$n@$($m.Groups[2].Value)" --precise $target 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $fixes += "$n $($m.Groups[2].Value)->$target"; break }
        }
    }
    return $fixes
}

# 2. Test-Units bestimmen (Repos ohne Root-Cargo.toml -> Unterordner einzeln)
$repos = Get-ChildItem $corpus -Directory
$units = @()
foreach ($r in $repos) {
    if (Test-Path (Join-Path $r.FullName "Cargo.toml")) { $units += $r }
    else {
        $units += Get-ChildItem $r.FullName -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName "Cargo.toml")
        }
    }
}

$csv = Join-Path $resultsDir "results.csv"
"repo,cargo_exit,selected_pkgs,analyzed_crates,execute_roots,ice_or_panic,remediations,duration_s" | Out-File $csv -Encoding utf8

foreach ($r in $units) {
    $name = if ($r.Parent.FullName -eq $corpus) { $r.Name } else { "$($r.Parent.Name)__$($r.Name)" }
    if ($Repo -ne "" -and $name -ne $Repo) { continue }
    Write-Host "`n== Teste $name =="
    $log = Join-Path $resultsDir "$name.log"
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()

    Push-Location $r.FullName
    $env:RUSTUP_TOOLCHAIN = $toolchain
    $env:CARGO_TARGET_DIR = Join-Path $resultsDir "target\$name"

    # 2a. Lockfile-Reparatur
    $fixes = Repair-Lockfile $r.FullName
    if ($fixes.Count) { Write-Host "   Lockfile-Fixes: $($fixes -join '; ')" }

    # 2b. Contract-Crates = Workspace-Member mit cdylib-Target
    $meta = rustup run $toolchain cargo metadata --format-version 1 --no-deps 2>$null | ConvertFrom-Json
    $pkgs = @($meta.packages | Where-Object {
        ($_.targets | Where-Object { $_.kind -contains "cdylib" })
    })

    if ($pkgs.Count -eq 0) {
        "$name,SKIP,0,0,0,no,`"keine cdylib-Crates`",0" | Add-Content $csv
        Remove-Item Env:RUSTUP_TOOLCHAIN, Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
        Pop-Location; continue
    }
    # volle Package-ID-Specs (path+file://...#name@version): Workspace-Member
    # können mit gleichnamigen UND versionsgleichen Dependencies kollidieren
    $pArgs = @()
    foreach ($p in $pkgs) { $pArgs += "-p"; $pArgs += $p.id }

    # 2c. Check mit Airlock als Wrapper
    $env:RUSTC_WRAPPER = $airlock
    rustup run $toolchain cargo check @pArgs --quiet 2>&1 |
        ForEach-Object { "$_" } | Out-File $log -Encoding utf8
    $exit = $LASTEXITCODE
    $out  = Get-Content $log -Raw

    Remove-Item Env:RUSTC_WRAPPER, Env:RUSTUP_TOOLCHAIN, Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
    Pop-Location
    $sw.Stop()

    $analyzed = ([regex]::Matches($out, "Analysiere Contract-Crate|Analyze Contract-Crate")).Count
    $roots    = ([regex]::Matches($out, "Found Execute Entry Point:")).Count
    # nur echte Compiler-/Airlock-Panics zählen, keine Build-Script-Panics
    $ice      = if ($out -match "internal compiler error|thread 'rustc' panicked|airlock(\.exe)? panicked") { "yes" } else { "no" }

    "$name,$exit,$($pkgs.Count),$analyzed,$roots,$ice,`"$($fixes -join '; ')`",$([int]$sw.Elapsed.TotalSeconds)" | Add-Content $csv
    Write-Host "   exit=$exit pkgs=$($pkgs.Count) analysiert=$analyzed roots=$roots panic=$ice"
}

Write-Host "`nFertig. Ergebnisse: $csv"
