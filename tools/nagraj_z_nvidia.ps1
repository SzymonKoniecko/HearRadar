<#
.SYNOPSIS
Nagrywa sesję radaru (--recordSample z sygnałem synchronizacji) i równolegle włącza nagrywanie NVIDIA skrótem Alt+F9.

.DESCRIPTION
Najpierw wysyła Alt+F9 (start nagrywania w NVIDIA), potem uruchamia radar z --recordSample --sygnal.
Radar na starcie odtwarza pół sekundy tonu 1 kHz na CABLE Input. Po zakończeniu radaru wysyła Alt+F9
jeszcze raz (stop nagrywania NVIDIA).
Wymaga: włączonego nakładania NVIDIA z domyślnym skrótem Alt+F9 oraz CABLE Input jako domyślnego
wyjścia, bo NVIDIA nagrywa to, co idzie na domyślne wyjście. Bez tego nie złapie sygnału synchronizacji.

.EXAMPLE
.\tools\nagraj_z_nvidia.ps1 -Sesja sesja_test -Sekundy 60 -Gra "Gra X"
#>
param(
    [Parameter(Mandatory = $true)] [string] $Sesja,
    [int] $Sekundy = 60,
    [string] $Gra = "",
    [int] $OpoznienieNvidiaMs = 1500,
    [string] $Radar = ""
)

Add-Type -AssemblyName System.Windows.Forms

function Przelacz-NvidiaAltF9 {
    [System.Windows.Forms.SendKeys]::SendWait("%{F9}")
}

# Domyślna ścieżka liczona w ciele skryptu: w bloku param $PSScriptRoot bywa pusty.
if (-not $Radar) {
    $katalogSkryptu = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $Radar = Join-Path $katalogSkryptu "..\target\release\radar.exe"
}

if (-not (Test-Path -LiteralPath $Radar)) {
    Write-Error "Brak radar.exe: $Radar"
    exit 2
}

$radarArgs = @("--recordSample", $Sesja, "--seconds", "$Sekundy", "--sygnal")
if ($Gra) {
    $radarArgs += @("--game", $Gra)
}

Write-Host "Start nagrywania NVIDIA (Alt+F9)"
Przelacz-NvidiaAltF9
Start-Sleep -Milliseconds $OpoznienieNvidiaMs
$kod = 0
try {
    & $Radar @radarArgs
    $kod = $LASTEXITCODE
} finally {
    Start-Sleep -Milliseconds 500
    Write-Host "Stop nagrywania NVIDIA (Alt+F9)"
    Przelacz-NvidiaAltF9
}
exit $kod
