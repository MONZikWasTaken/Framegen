# Assemble the self-contained Chrome extension: runtime + weights into extension/.
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\extension\rt" | Out-Null
New-Item -ItemType Directory -Force "$root\extension\assets" | Out-Null
Copy-Item "$root\web\rt\rt.js" "$root\extension\rt\rt.js" -Force
Copy-Item "$root\web\rt\sr.js" "$root\extension\rt\sr.js" -Force
Copy-Item "$root\assets\rt_tfact.bin" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_tfact.json" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_sr.bin" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_sr.json" "$root\extension\assets\" -Force
# slim copies no longer shipped — tfact replaced them
Remove-Item "$root\extension\assets\rt_slim.bin", "$root\extension\assets\rt_slim.json" -Force -ErrorAction SilentlyContinue
$size = (Get-ChildItem "$root\extension" -Recurse | Measure-Object Length -Sum).Sum
Write-Host ("extension ready: {0:N1} MB" -f ($size / 1MB))
# distributable zip (load-unpacked-able after extraction; also Web-Store-uploadable).
# Chrome writes _metadata/ into the source dir when the unpacked extension has DNR
# rulesets — shipping it makes Chrome REFUSE to load ("_ names are reserved"): stage
# a filtered copy first.
$stage = Join-Path $env:TEMP "framecast-zip-stage"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null
Get-ChildItem "$root\extension" | Where-Object { $_.Name -notlike '_*' } |
    ForEach-Object { Copy-Item $_.FullName (Join-Path $stage $_.Name) -Recurse -Force }
$zip = "$root\framecast-extension.zip"
Compress-Archive -Path "$stage\*" -DestinationPath $zip -Force
Write-Host ("zip: {0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
