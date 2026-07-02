# Assemble the self-contained Chrome extension: runtime + weights into extension/.
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\extension\rt" | Out-Null
New-Item -ItemType Directory -Force "$root\extension\assets" | Out-Null
Copy-Item "$root\web\rt\rt.js" "$root\extension\rt\rt.js" -Force
Copy-Item "$root\assets\rt_slim.bin" "$root\extension\assets\" -Force
Copy-Item "$root\assets\rt_slim.json" "$root\extension\assets\" -Force
$size = (Get-ChildItem "$root\extension" -Recurse | Measure-Object Length -Sum).Sum
Write-Host ("extension ready: {0:N1} MB" -f ($size / 1MB))
