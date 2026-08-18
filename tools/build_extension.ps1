[CmdletBinding()]
param([switch] $PromoteLocalAssets)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Assemble the self-contained Chrome extension. The tracked extension/ tree is
# directly loadable; this script refreshes the runtime mirror, validates every
# bundled model, and creates both release archives.
$root = (Resolve-Path (Split-Path $PSScriptRoot -Parent)).Path
$extensionRoot = Join-Path $root 'extension'
$runtimeRoot = Join-Path $extensionRoot 'rt'
$assetRoot = Join-Path $extensionRoot 'assets'
New-Item -ItemType Directory -Force -Path $runtimeRoot, $assetRoot | Out-Null

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

foreach ($runtimeName in @('rt.js', 'sr.js')) {
    $source = Join-Path (Join-Path $root 'web/rt') $runtimeName
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing runtime source: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $runtimeRoot $runtimeName) -Force
}
Assert-ExactFiles -Directory $runtimeRoot -AllowedNames @('rt.js', 'sr.js')

function Assert-ModelBundle {
    param(
        [Parameter(Mandatory)][string] $Stem,
        [Parameter(Mandatory)][string] $Directory
    )

    $binPath = Join-Path $Directory "$Stem.bin"
    $manifestPath = Join-Path $Directory "$Stem.json"
    if (!(Test-Path -LiteralPath $binPath -PathType Leaf)) {
        throw "Missing bundled model weights: $binPath"
    }
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Missing bundled model manifest: $manifestPath"
    }

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
        $offsetNumber = [double]$offsetProperty.Value
        if ([double]::IsNaN($offsetNumber) -or [double]::IsInfinity($offsetNumber) -or
            $offsetNumber -lt 0 -or $offsetNumber -ne [math]::Truncate($offsetNumber)) {
            throw "Invalid tensor offset for '$($property.Name)' in $manifestPath"
        }
        $offset = [long]$offsetNumber
        $shape = @($shapeProperty.Value)
        if ($shape.Count -eq 0) {
            throw "Empty tensor shape for '$($property.Name)' in $manifestPath"
        }
        $elements = [long]1
        foreach ($dimensionValue in $shape) {
            $dimensionNumber = [double]$dimensionValue
            if ([double]::IsNaN($dimensionNumber) -or [double]::IsInfinity($dimensionNumber) -or
                $dimensionNumber -le 0 -or $dimensionNumber -ne [math]::Truncate($dimensionNumber)) {
                throw "Invalid tensor shape for '$($property.Name)' in $manifestPath"
            }
            $dimension = [long]$dimensionNumber
            if ($elements -gt [long]::MaxValue / $dimension) {
                throw "Tensor shape overflow for '$($property.Name)' in $manifestPath"
            }
            $elements *= $dimension
        }
        if ($offset -gt [long]::MaxValue - $elements) {
            throw "Tensor range overflow for '$($property.Name)' in $manifestPath"
        }
        $ranges += [pscustomobject]@{ Name = $property.Name; Offset = $offset; End = $offset + $elements }
    }
    if ($ranges.Count -eq 0) {
        throw "Empty model manifest: $manifestPath"
    }

    $cursor = [long]0
    foreach ($range in ($ranges | Sort-Object Offset)) {
        if ($range.Offset -ne $cursor) {
            throw "Non-contiguous tensor '$($range.Name)' in $manifestPath (expected offset $cursor, got $($range.Offset))"
        }
        $cursor = $range.End
    }
    if ($cursor -gt [long]::MaxValue / 4) {
        throw "Model byte size overflow for $Stem"
    }
    $expectedBytes = $cursor * 4
    $actualBytes = (Get-Item -LiteralPath $binPath).Length
    if ($actualBytes -ne $expectedBytes) {
        throw "Model size mismatch for $Stem (expected $expectedBytes bytes, got $actualBytes)"
    }
}

$modelStems = @('rt_tfact2', 'rt_v7s', 'rt_sr')
$localAssetRoot = Join-Path $root 'assets'
$localModelStems = @()
foreach ($stem in $modelStems) {
    $localBin = Join-Path $localAssetRoot "$stem.bin"
    $localManifest = Join-Path $localAssetRoot "$stem.json"
    $hasLocalBin = Test-Path -LiteralPath $localBin -PathType Leaf
    $hasLocalManifest = Test-Path -LiteralPath $localManifest -PathType Leaf
    if ($hasLocalBin -ne $hasLocalManifest) {
        throw "Incomplete local model export for $stem; both .bin and .json are required"
    }
    if (!$hasLocalBin) {
        continue
    }
    $localModelStems += $stem

    $bundledBin = Join-Path $assetRoot "$stem.bin"
    $bundledManifest = Join-Path $assetRoot "$stem.json"
    if ($PromoteLocalAssets) {
        continue
    }

    $filePairs = @(
        [pscustomobject]@{ Local = $localBin; Bundled = $bundledBin },
        [pscustomobject]@{ Local = $localManifest; Bundled = $bundledManifest }
    )
    foreach ($pair in $filePairs) {
        if (!(Test-Path -LiteralPath $pair.Bundled -PathType Leaf) -or
            (Get-FileHash -Algorithm SHA256 -LiteralPath $pair.Local).Hash -ne
            (Get-FileHash -Algorithm SHA256 -LiteralPath $pair.Bundled).Hash) {
            throw "Local model export for $stem differs from extension/assets; rerun with -PromoteLocalAssets and commit the resulting payload"
        }
    }
}

if ($PromoteLocalAssets) {
    # Validate every available export before modifying the tracked payload so a
    # bad later model cannot leave an earlier one partially promoted.
    foreach ($stem in $localModelStems) {
        Assert-ModelBundle -Stem $stem -Directory $localAssetRoot
    }
    foreach ($stem in $localModelStems) {
        Copy-Item -LiteralPath (Join-Path $localAssetRoot "$stem.bin") -Destination (Join-Path $assetRoot "$stem.bin") -Force
        Copy-Item -LiteralPath (Join-Path $localAssetRoot "$stem.json") -Destination (Join-Path $assetRoot "$stem.json") -Force
    }
}

Assert-ExactFiles -Directory $assetRoot -AllowedNames @(
    'rt_tfact2.bin', 'rt_tfact2.json',
    'rt_v7s.bin', 'rt_v7s.json',
    'rt_sr.bin', 'rt_sr.json'
)
foreach ($stem in $modelStems) {
    Assert-ModelBundle -Stem $stem -Directory $assetRoot
}

$size = (Get-ChildItem -LiteralPath $extensionRoot -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("extension ready: {0:N1} MB" -f ($size / 1MB))

# Chrome writes _metadata/ into unpacked extensions. Stage a filtered copy so
# neither distributable contains reserved browser-owned entries.
$tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$stage = Join-Path $tempBase ('framegen-zip-stage-' + [guid]::NewGuid().ToString('N'))
$stage = [System.IO.Path]::GetFullPath($stage)
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
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}
