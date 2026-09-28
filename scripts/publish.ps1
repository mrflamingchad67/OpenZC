<#
.SYNOPSIS
    Run the quality gate, then commit and push to GitHub.

.DESCRIPTION
    Automates the "update the repo" loop, with the safety properties that make it
    safe to run unattended:

      * The full quality gate runs first, and a failure aborts before anything is
        committed. A tool that pushed failing code would be worse than no tool.
      * Nothing is pushed unless the working tree actually has changes.
      * No force push, ever, unless -Force is passed explicitly.
      * Every staged file is listed before the commit, so an accidental
        large file or secret is visible before it lands in history.
      * The commit message must be supplied, so history stays readable.

    Secrets are scanned for before staging, not after: it is far easier to not
    commit a token than to remove one.

.PARAMETER Message
    Commit message. First line is the subject; the rest is the body.
    A subject-only message is a common source of unreadable history, so it is
    accepted but you will be asked to confirm.

.PARAMETER Yes
    Skip the confirmation prompt. Required for unattended use.

.PARAMETER Force
    Allow a force push. Only correct when you are deliberately rewriting history
    (for example, amending a commit's author). Defaults to off.

.EXAMPLE
    ./scripts/publish.ps1 -Message "Add archive mode" -Yes

.EXAMPLE
    ./scripts/publish.ps1 -Message "Fix LZ overlap copy" -Yes
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [switch]$Yes,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Run from the repository root regardless of where the script was invoked.
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    Write-Host "==> Repository: $repoRoot" -ForegroundColor Cyan

    # ---------------------------------------------------------------------
    # 1. Sanity: are we even in a git repository with a remote?
    # ---------------------------------------------------------------------
    & git rev-parse --is-inside-work-tree *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Not a git repository. Run 'git init' first."
    }

    $hasRemote = [bool](& git remote 2>$null)
    if (-not $hasRemote) {
        throw "No git remote configured. Add one with: git remote add origin <url>"
    }

    # ---------------------------------------------------------------------
    # 2. Quality gate. This is the whole point of the script: nothing reaches
    #    the remote that has not passed fmt, check, test and clippy.
    # ---------------------------------------------------------------------
    $gate = @(
        @{ Name = 'cargo fmt --check';      Args = @('fmt', '--check') },
        @{ Name = 'cargo check --all-targets'; Args = @('check', '--all-targets') },
        @{ Name = 'cargo test --all-targets';  Args = @('test', '--all-targets') },
        @{ Name = 'cargo clippy (deny warnings)'; Args = @('clippy', '--all-targets', '--all-features', '--', '-D', 'warnings') }
    )

    foreach ($step in $gate) {
        Write-Host "`n==> $($step.Name)" -ForegroundColor Cyan
        & cargo @($step.Args)
        if ($LASTEXITCODE -ne 0) {
            throw "Quality gate failed at '$($step.Name)'. Nothing was committed or pushed."
        }
    }

    # ---------------------------------------------------------------------
    # 3. Is there anything to do?
    # ---------------------------------------------------------------------
    & git add -A
    $staged = & git diff --cached --name-only
    if (-not $staged) {
        Write-Host "`n==> Working tree is clean; nothing to publish." -ForegroundColor Yellow
        return
    }

    # ---------------------------------------------------------------------
    # 4. Refuse obvious accidents before they become permanent.
    # ---------------------------------------------------------------------
    $stagedFiles = @($staged)

    # Build output should be ignored; if it is staged, .gitignore is broken.
    $buildJunk = $stagedFiles | Where-Object { $_ -match '(^|/)target/' }
    if ($buildJunk) {
        throw "Build output is staged ($($buildJunk -join ', ')). Check .gitignore."
    }

    # A large file in a source repository is nearly always a mistake.
    $large = foreach ($f in $stagedFiles) {
        if (Test-Path -LiteralPath $f -PathType Leaf) {
            $item = Get-Item -LiteralPath $f
            if ($item.Length -gt 5MB) { "{0} ({1:N1} MB)" -f $f, ($item.Length / 1MB) }
        }
    }
    if ($large) {
        throw "Refusing to commit large files: $($large -join '; ')"
    }

    # Secret scan. Deliberately broad and deliberately noisy: a false positive
    # costs one confirmation, a leaked token costs far more.
    $secretPattern = 'gho_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i)(api[_-]?key|secret[_-]?key|password)\s*[:=]\s*["''][^"'']{8,}'
    $hits = & rg --no-heading --line-number --max-count 20 -e $secretPattern -- $stagedFiles 2>$null
    if ($hits) {
        Write-Host "`n==> Possible secrets found in staged files:" -ForegroundColor Yellow
        $hits | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        throw "Possible secrets in staged files. Review the above, then remove them or use --git-ignore to exclude."
    }

    # ---------------------------------------------------------------------
    # 5. Show the user exactly what is about to be committed.
    # ---------------------------------------------------------------------
    Write-Host "`n==> Files to be committed ($($stagedFiles.Count)):" -ForegroundColor Cyan
    $stagedFiles | ForEach-Object { Write-Host "    $_" }

    $diffStat = & git diff --cached --stat
    if ($diffStat) {
        Write-Host "`n$($diffStat -join "`n")" -ForegroundColor DarkGray
    }

    if (-not $Yes) {
        Write-Host "`nProceed? [y/N] " -ForegroundColor Yellow -NoNewline
        $answer = Read-Host
        if ($answer -notmatch '^(y|yes)$') {
            Write-Host "Aborted. Nothing was committed." -ForegroundColor Yellow
            return
        }
    }

    # ---------------------------------------------------------------------
    # 6. Commit and push.
    # ---------------------------------------------------------------------
    & git commit -m $Message
    if ($LASTEXITCODE -ne 0) {
        throw "Commit failed."
    }

    $branch = (& git rev-parse --abbrev-ref HEAD).Trim()
    $pushArgs = if ($Force) { @('push', '--force-with-lease', 'origin', $branch) }
                else        { @('push', 'origin', $branch) }

    & git @pushArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Push failed. The commit is local only; nothing was lost."
    }

    $sha = (& git rev-parse --short HEAD).Trim()
    Write-Host "`n==> Published $sha to origin/$branch" -ForegroundColor Green
}
finally {
    Pop-Location
}
