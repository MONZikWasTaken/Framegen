# Assemble the self-contained Chrome extension: runtime + weights into extension/.
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\extension\rt" | Out-Null
New-Item -ItemType Directory -Force "$root\extension\assets" | Out-Null
Copy-Item "$root\web\rt\rt.js" "$root\extension\rt\rt.js" -Force
Copy-Item "$root\web\rt\sr.js" "$root\extension\rt\sr.js" -Force
Copy-Item "$root\assets\rt_slim.bin" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_slim.json" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_sr.bin" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_sr.json" "$root\extension\assets\" -Force
$size = (Get-ChildItem "$root\extension" -Recurse | Measure-Object Length -Sum).Sum
Write-Host ("extension ready: {0:N1} MB" -f ($size / 1MB))
# distributable zip (load-unpacked-able after extraction; also Web-Store-uploadable)
$zip = "$root\framecast-extension.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$root\extension\*" -DestinationPath $zip
Write-Host ("zip: {0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
