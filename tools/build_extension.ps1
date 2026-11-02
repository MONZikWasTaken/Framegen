[CmdletBinding()]
param(
    # A checkout of the FlyHigh engine: its browser module (examples/web/pkg, built by tools/build_web.sh), the adapters and the
    # model files are copied into extension/engine before validation.
    [string] $EngineRepo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Assemble the extension. The tracked extension/ tree is directly loadable; this script validates every bundled
# payload and creates the release archives for Chrome, the Web Store and Firefox.
$root = (Resolve-Path (Split-Path $PSScriptRoot -Parent)).Path
$extensionRoot = Join-Path $root 'extension'
$engineRoot = Join-Path $extensionRoot 'engine'
$runtimeRoot = Join-Path $extensionRoot 'rt'
$assetRoot = Join-Path $extensionRoot 'assets'
$sourceRoot = Join-Path $extensionRoot 'src'

function Assert-ExactFiles {
    param(
        [Parameter(Mandatory)][string] $Directory,
        [Parameter(Mandatory)][string[]] $AllowedNames
    )

    foreach ($name in $AllowedNames) {
        $path = Join-Path $Directory $name
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing required extension file: $path"
        }
    }
    $unexpected = @(Get-ChildItem -LiteralPath $Directory | Where-Object {
        $_.PSIsContainer -or $_.Name -notin $AllowedNames
    })
    if ($unexpected.Count) {
        throw "Unexpected extension files in ${Directory}: $($unexpected.Name -join ', ')"
    }
}

if ($EngineRepo) {
    $engine = (Resolve-Path $EngineRepo).Path
    $copies = @(
        @{ From = 'examples/web/pkg/flyhigh_web.js'; To = 'flyhigh_web.js' },
        @{ From = 'examples/web/pkg/flyhigh_web_bg.wasm'; To = 'flyhigh_web_bg.wasm' },
        @{ From = 'crates/flyhigh-web/js/adapters.js'; To = 'adapters.js' },
        @{ From = 'models/graph.json'; To = 'graph.json' },
        @{ From = 'models/flyhighfg.json'; To = 'flyhighfg.json' },
        @{ From = 'models/flyhighfg.bin'; To = 'flyhighfg.bin' },
        @{ From = 'models/WEIGHTS_LICENSE.md'; To = 'WEIGHTS_LICENSE.md' }
    )
    foreach ($copy in $copies) {
        $source = Join-Path $engine $copy.From
        if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing engine file: $source" }
        Copy-Item -LiteralPath $source -Destination (Join-Path $engineRoot $copy.To) -Force
    }
    Write-Host "engine payload refreshed from $engine"
}

# The FlyHigh engine: the wasm-bindgen module, the browser adapters, the model (graph, manifest, f16 weights).
Assert-ExactFiles -Directory $engineRoot -AllowedNames @(
    'flyhigh_web.js', 'flyhigh_web_bg.wasm', 'adapters.js',
    'graph.json', 'flyhighfg.json', 'flyhighfg.bin', 'WEIGHTS_LICENSE.md'
)
$wasmHeader = [System.IO.File]::ReadAllBytes((Join-Path $engineRoot 'flyhigh_web_bg.wasm'))[0..3]
if (($wasmHeader -join ',') -ne '0,97,115,109') { throw 'flyhigh_web_bg.wasm is not a WebAssembly module' }
$flyManifest = Get-Content -Raw -LiteralPath (Join-Path $engineRoot 'flyhighfg.json') | ConvertFrom-Json
$flyTensors = $flyManifest.PSObject.Properties['tensors']
if ($null -eq $flyTensors) { throw "flyhighfg.json has no 'tensors' table" }
$flyEnd = [long]0
foreach ($tensorProperty in $flyTensors.Value.PSObject.Properties) {
    $entry = $tensorProperty.Value
    $elements = [long]1
    foreach ($dimensionValue in @($entry.shape)) { $elements *= [long]$dimensionValue }
    $end = [long]$entry.offset + $elements
    if ($end -gt $flyEnd) { $flyEnd = $end }
}
$flyExpectedBytes = $flyEnd * 2   # little-endian f16
$flyActualBytes = (Get-Item -LiteralPath (Join-Path $engineRoot 'flyhighfg.bin')).Length
if ($flyActualBytes -ne $flyExpectedBytes) {
    throw "FlyHigh weights size mismatch (expected $flyExpectedBytes bytes, got $flyActualBytes)"
}
$moduleHead = Get-Content -TotalCount 40 -LiteralPath (Join-Path $engineRoot 'flyhigh_web.js') | Out-String
if ($moduleHead -notmatch 'FhEngine|export') { throw 'flyhigh_web.js does not look like the wasm-bindgen module' }

# TinySR, the upscaler: sr.js with its conv kernels in conv.js, its weights in assets/.
Assert-ExactFiles -Directory $runtimeRoot -AllowedNames @('conv.js', 'sr.js')

function Assert-ModelBundle {
    param(
        [Parameter(Mandatory)][string] $Stem,
        [Parameter(Mandatory)][string] $Directory
    )

    $binPath = Join-Path $Directory "$Stem.bin"
    $manifestPath = Join-Path $Directory "$Stem.json"
    if (!(Test-Path -LiteralPath $binPath -PathType Leaf)) { throw "Missing bundled model weights: $binPath" }
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Missing bundled model manifest: $manifestPath" }

    try {
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    } catch {
        throw "Invalid JSON model manifest '$manifestPath': $($_.Exception.Message)"
    }

    $ranges = @()
    foreach ($property in $manifest.PSObject.Properties) {
        $tensor = $property.Value
        $offsetProperty = $tensor.PSObject.Properties['offset']
        $shapeProperty = $tensor.PSObject.Properties['shape']
        if ($null -eq $offsetProperty -or $null -eq $shapeProperty) {
            throw "Invalid tensor entry '$($property.Name)' in $manifestPath"
        }
        $offset = [long]$offsetProperty.Value
        $elements = [long]1
        foreach ($dimensionValue in @($shapeProperty.Value)) { $elements *= [long]$dimensionValue }
        $ranges += [pscustomobject]@{ Name = $property.Name; Offset = $offset; End = $offset + $elements }
    }
    if ($ranges.Count -eq 0) { throw "Empty model manifest: $manifestPath" }

    $cursor = [long]0
    foreach ($range in ($ranges | Sort-Object Offset)) {
        if ($range.Offset -ne $cursor) {
            throw "Non-contiguous tensor '$($range.Name)' in $manifestPath (expected offset $cursor, got $($range.Offset))"
        }
        $cursor = $range.End
    }
    $expectedBytes = $cursor * 4
    $actualBytes = (Get-Item -LiteralPath $binPath).Length
    if ($actualBytes -ne $expectedBytes) {
        throw "Model size mismatch for $Stem (expected $expectedBytes bytes, got $actualBytes)"
    }
}

Assert-ExactFiles -Directory $assetRoot -AllowedNames @('rt_sr.bin', 'rt_sr.json')
foreach ($stem in @('rt_sr')) {
    Assert-ModelBundle -Stem $stem -Directory $assetRoot
}

# Every module the content script imports must be a web-accessible file.
$manifest = Get-Content -Raw -LiteralPath (Join-Path $extensionRoot 'manifest.json') | ConvertFrom-Json
$accessible = @($manifest.web_accessible_resources | ForEach-Object { $_.resources })
foreach ($prefix in @('src/*', 'engine/*', 'rt/*', 'assets/*')) {
    if ($prefix -notin $accessible) { throw "manifest.json web_accessible_resources lacks $prefix" }
}
foreach ($module in Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter '*.js') {
    $text = Get-Content -Raw -LiteralPath $module.FullName
    foreach ($match in [regex]::Matches($text, "from '(\./[^']+)'")) {
        $target = Join-Path $module.DirectoryName $match.Groups[1].Value.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        if (!(Test-Path -LiteralPath $target -PathType Leaf)) {
            throw "$($module.Name) imports a missing module: $($match.Groups[1].Value)"
        }
    }
}

# AMO rejects archives whose entry names use '\' (what Compress-Archive writes on Windows PowerShell 5.1)
function New-ForwardSlashZip {
    param(
        [Parameter(Mandatory)][string] $SourceDir,
        [Parameter(Mandatory)][string] $ZipPath
    )

    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
    $archive = [System.IO.Compression.ZipFile]::Open($ZipPath, 'Create')
    try {
        $sep = [System.IO.Path]::DirectorySeparatorChar
        $base = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd($sep) + $sep
        foreach ($file in Get-ChildItem -LiteralPath $SourceDir -Recurse -File) {
            $entryName = $file.FullName.Substring($base.Length).Replace($sep, '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $entryName) | Out-Null
        }
    } finally {
        $archive.Dispose()
    }
}

$size = (Get-ChildItem -LiteralPath $extensionRoot -Recurse -File | Where-Object { $_.FullName -notmatch '[\\/]_metadata[\\/]' } | Measure-Object Length -Sum).Sum
Write-Host ("extension ready: {0:N1} MB" -f ($size / 1MB))

# Chrome writes _metadata/ into unpacked extensions; the archives are made from a filtered copy.
$tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$stage = [System.IO.Path]::GetFullPath((Join-Path $tempBase ('framegen-zip-stage-' + [guid]::NewGuid().ToString('N'))))
if (!$stage.StartsWith($tempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $stage"
}

try {
    $stageDir = Join-Path $stage 'framegen-extension'
    New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
    Get-ChildItem -LiteralPath $extensionRoot | Where-Object { $_.Name -notlike '_*' } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stageDir $_.Name) -Recurse -Force
        }

    $zip = Join-Path $root 'framegen-extension.zip'
    Compress-Archive -Path $stageDir -DestinationPath $zip -Force
    Write-Host ("zip: {0} ({1:N1} MB)" -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB))

    $storeZip = Join-Path $root 'framegen-webstore.zip'
    Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $storeZip -Force
    Write-Host ("store zip: {0} ({1:N1} MB)" -f $storeZip, ((Get-Item -LiteralPath $storeZip).Length / 1MB))

    # Firefox build: the same tree with Gecko manifest keys: an event-page background (no MV3 service worker there),
    # the add-on id AMO signs, no Chrome-only version_name.
    $firefoxDir = Join-Path $stage 'framegen-firefox'
    Copy-Item -LiteralPath $stageDir -Destination $firefoxDir -Recurse -Force
    $firefoxManifestPath = Join-Path $firefoxDir 'manifest.json'
    $firefoxManifest = Get-Content -Raw -LiteralPath $firefoxManifestPath | ConvertFrom-Json
    $firefoxManifest.background = [pscustomobject]@{ scripts = @('background.js') }
    $firefoxManifest.PSObject.Properties.Remove('version_name')
    $firefoxManifest | Add-Member -Force -NotePropertyName browser_specific_settings -NotePropertyValue (
        [pscustomobject]@{ gecko = [pscustomobject]@{ id = 'framegen@flyhigh.dev'; strict_min_version = '141.0' } })
    [System.IO.File]::WriteAllText($firefoxManifestPath, ($firefoxManifest | ConvertTo-Json -Depth 20),
        (New-Object System.Text.UTF8Encoding $false))
    $firefoxZip = Join-Path $root 'framegen-firefox.zip'
    New-ForwardSlashZip -SourceDir $firefoxDir -ZipPath $firefoxZip
    Write-Host ("firefox zip: {0} ({1:N1} MB)" -f $firefoxZip, ((Get-Item -LiteralPath $firefoxZip).Length / 1MB))
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}
