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

Write-Host "== HearRadar: cold start klasyfikacji ML (Windows) =="

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Error "Nie znaleziono Pythona w PATH. Zainstaluj Python 3.9+ z https://python.org (zaznacz 'Add python.exe to PATH') i uruchom ten skrypt ponownie."
    exit 1
}
$pythonExe = $python.Source

if (-not (Test-Path ".venv")) {
    Write-Host "-> tworze wirtualne srodowisko (.venv)"
    & $pythonExe -m venv .venv
}

$venvPython = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Nie znaleziono $venvPython -- tworzenie .venv nie powiodlo sie."
    exit 1
}

Write-Host "-> instaluje zaleznosci (ml/requirements.txt)"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt --quiet

New-Item -ItemType Directory -Force -Path "out" | Out-Null

Write-Host "-> eksportuje YAMNet do ONNX"
& $venvPython export_yamnet_onnx.py

Write-Host "-> buduje mapowanie 521 klas AudioSet -> kategorie"
& $venvPython export_mapping.py

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
