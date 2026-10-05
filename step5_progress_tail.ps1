param(
    [Parameter(Mandatory=$true)][string]$LogPath,
    [Parameter(Mandatory=$true)][string]$DonePath,
    [int]$PollMilliseconds = 250
)

$ErrorActionPreference = 'SilentlyContinue'
$seen = 0

while ($true) {
    if (Test-Path -LiteralPath $LogPath) {
        $lines = @(Get-Content -LiteralPath $LogPath)
        if ($lines.Count -gt $seen) {
            for ($i = $seen; $i -lt $lines.Count; $i++) {
                [Console]::Out.WriteLine($lines[$i])
            }
            $seen = $lines.Count
        }
    }

    if (Test-Path -LiteralPath $DonePath) {
        break
    }

    Start-Sleep -Milliseconds $PollMilliseconds
}
