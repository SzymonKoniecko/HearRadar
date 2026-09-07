# Cold start dla klasyfikacji dzwieku ML na Windows -- odpowiednik
# ml/bootstrap.sh (macOS/Linux). Jedna komenda, ktora przygotowuje wszystko
# potrzebne, zeby wlaczyc eksperymentalna klasyfikacje w HearRadar.
#
#   .\ml\bootstrap.ps1
#
# Jesli PowerShell odmawia uruchomienia skryptu (domyslna polityka
# wykonywania), uruchom to raz w PowerShell z uprawnieniami administratora
# (albo tylko dla biezacego uzytkownika):
#
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# ...albo jednorazowo, bez zmiany polityki na stale:
#
#   powershell -ExecutionPolicy Bypass -File ml\bootstrap.ps1
#
# Wymaga Pythona 3.9+ z dostepem do internetu (pobiera YAMNet z TF-Hub oraz
# zaleznosci z PyPI) -- zajmuje kilka minut przy pierwszym uruchomieniu.
# Wynik: assets\ml\yamnet.onnx + assets\ml\category_mapping.json.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Znany problem na Windows: `python -m venv` wewnetrznie odpala NOWY proces
# pythona, zeby zainstalowac pip (ensurepip) -- ten podproces potrafi sie
# wywalic w trakcie czytania jego stdout, najczesciej przez:
#   a) konsole/locale inna niz UTF-8 (ensurepip/pip mieszaja sie na
#      niestandardowym kodowaniu wyjscia) -- to najczestsza przyczyna,
#   b) antywirusa/Windows Defender blokujacego swiezo odpalony python.exe.
# PYTHONUTF8=1 usuwa przyczyne (a) w zdecydowanej wiekszosci przypadkow.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "== HearRadar: cold start klasyfikacji ML (Windows) =="

# PowerShell NIE traktuje niezerowego kodu wyjscia programu zewnetrznego
# (jak python.exe) jako bledu przechwytywalnego przez try/catch -- trzeba to
# sprawdzic recznie po kazdym wywolaniu, inaczej fallbacki nizej nigdy by sie
# nie uruchomily mimo `$ErrorActionPreference = "Stop"`.
function Invoke-Checked {
    param([string]$Exe, [string[]]$Args)
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "'$Exe $($Args -join ' ')' zakonczylo sie kodem $LASTEXITCODE"
    }
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Error "Nie znaleziono Pythona w PATH. Zainstaluj Python 3.9+ z https://python.org (zaznacz 'Add python.exe to PATH') i uruchom ten skrypt ponownie."
    exit 1
}
$pythonExe = $python.Source

# Tworzenie venv, z fallbackiem gdy standardowy bootstrap pip (ensurepip)
# wewnatrz `python -m venv` sie wywali (patrz komentarz wyzej) -- wtedy
# tworzymy venv BEZ pip i doinstalowujemy je osobno, co omija ten sam
# podproces, ktory wczesniej padl.
function New-VenvWithPipFallback {
    if (Test-Path ".venv") {
        Write-Host "-> .venv juz istnieje, pomijam tworzenie"
        return
    }

    Write-Host "-> tworze wirtualne srodowisko (.venv)"
    try {
        Invoke-Checked $pythonExe @("-m", "venv", ".venv")
        return
    } catch {
        Write-Host "   (standardowe tworzenie .venv nie powiodlo sie -- najczesciej to problem z ensurepip/kodowaniem na Windows, nie z Twoim ustawieniem)"
        Write-Host "   -> probuje ponownie bez wbudowanego pip, i doinstaluje pip osobno..."
    }

    if (Test-Path ".venv") {
        Remove-Item -Recurse -Force ".venv"
    }
    Invoke-Checked $pythonExe @("-m", "venv", ".venv", "--without-pip")

    $venvPythonForPip = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"
    try {
        Invoke-Checked $venvPythonForPip @("-m", "ensurepip", "--upgrade")
    } catch {
        Write-Host "   ensurepip tez sie nie udal -- pobieram pip bezposrednio (get-pip.py)..."
        $getPipPath = Join-Path $env:TEMP "get-pip.py"
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath
        Invoke-Checked $venvPythonForPip @($getPipPath)
    }
}

New-VenvWithPipFallback

$venvPython = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Nie znaleziono $venvPython -- tworzenie .venv nie powiodlo sie mimo fallbackow. Sprawdz, czy antywirus nie blokuje uruchamiania python.exe, i sprobuj ponownie (ewentualnie usun katalog .venv recznie przed kolejna proba)."
    exit 1
}

Write-Host "-> instaluje zaleznosci (ml/requirements.txt)"
Invoke-Checked $venvPython @("-m", "pip", "install", "--upgrade", "pip", "--quiet")
Invoke-Checked $venvPython @("-m", "pip", "install", "-r", "requirements.txt", "--quiet")

New-Item -ItemType Directory -Force -Path "out" | Out-Null

Write-Host "-> eksportuje YAMNet do ONNX"
Invoke-Checked $venvPython @("export_yamnet_onnx.py")

Write-Host "-> buduje mapowanie 521 klas AudioSet -> kategorie"
Invoke-Checked $venvPython @("export_mapping.py")

$assetsDir = Join-Path ".." "assets\ml"
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null
Copy-Item "out\yamnet.onnx" (Join-Path $assetsDir "yamnet.onnx") -Force
Copy-Item "out\category_mapping.json" (Join-Path $assetsDir "category_mapping.json") -Force

Write-Host ""
Write-Host "Gotowe. Pliki modelu:"
Get-ChildItem $assetsDir | Format-Table Name, Length -AutoSize

Write-Host ""
Write-Host "Wlacz 'Klasyfikacje dzwieku (ML, eksperymentalne)' w ustawieniach radaru."
Write-Host "(Kwantyzacja INT8 jest opcjonalna -- patrz ml\quantize.py, jesli inferencja jest za wolna.)"
