# Reports or changes the primary display's refresh rate at its current resolution.
# Win32_VideoController.CurrentRefreshRate names only one attached panel and can
# report a secondary display, so the primary mode is read through
# EnumDisplayDevices + EnumDisplaySettings instead.
#
#   -Query            print the primary display's current refresh rate only
#   -Hz <rate>        apply <rate> to the primary display
#   -TestOnly         validate <rate> with CDS_TEST without applying it
[CmdletBinding()]
param(
  [int]$Hz = 0,
  [switch]$Query,
  [switch]$TestOnly
)

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class FgDisplay {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  private struct DEVMODE {
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
    public ushort dmSpecVersion;
    public ushort dmDriverVersion;
    public ushort dmSize;
    public ushort dmDriverExtra;
    public uint dmFields;
    public int dmPositionX;
    public int dmPositionY;
    public uint dmDisplayOrientation;
    public uint dmDisplayFixedOutput;
    public short dmColor;
    public short dmDuplex;
    public short dmYResolution;
    public short dmTTOption;
    public short dmCollate;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
    public ushort dmLogPixels;
    public uint dmBitsPerPel;
    public uint dmPelsWidth;
    public uint dmPelsHeight;
    public uint dmDisplayFlags;
    public uint dmDisplayFrequency;
    public uint dmICMMethod;
    public uint dmICMIntent;
    public uint dmMediaType;
    public uint dmDitherType;
    public uint dmReserved1;
    public uint dmReserved2;
    public uint dmPanningWidth;
    public uint dmPanningHeight;
  }

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  private struct DISPLAY_DEVICE {
    public uint cb;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string DeviceName;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceString;
    public uint StateFlags;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceID;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceKey;
  }

  [DllImport("user32.dll", CharSet = CharSet.Ansi)]
  private static extern bool EnumDisplaySettingsA(string dev, int mode, ref DEVMODE dm);

  [DllImport("user32.dll", CharSet = CharSet.Ansi)]
  private static extern int ChangeDisplaySettingsExA(string dev, ref DEVMODE dm, IntPtr hwnd, uint flags, IntPtr lp);

  [DllImport("user32.dll", CharSet = CharSet.Ansi)]
  private static extern bool EnumDisplayDevicesA(string dev, uint num, ref DISPLAY_DEVICE dd, uint flags);

  private const int ENUM_CURRENT_SETTINGS = -1;
  private const uint CDS_TEST = 0x02;
  private const uint ATTACHED_TO_DESKTOP = 0x1;
  private const uint PRIMARY_DEVICE = 0x4;

  private static DEVMODE NewDevMode() {
    DEVMODE dm = new DEVMODE();
    dm.dmSize = (ushort)Marshal.SizeOf(typeof(DEVMODE));
    return dm;
  }

  private static string PrimaryName() {
    for (uint i = 0; i < 16; i++) {
      DISPLAY_DEVICE dd = new DISPLAY_DEVICE();
      dd.cb = (uint)Marshal.SizeOf(typeof(DISPLAY_DEVICE));
      if (!EnumDisplayDevicesA(null, i, ref dd, 0)) break;
      if ((dd.StateFlags & ATTACHED_TO_DESKTOP) != 0 && (dd.StateFlags & PRIMARY_DEVICE) != 0) {
        return dd.DeviceName;
      }
    }
    return null;
  }

  public static string Current() {
    string name = PrimaryName();
    if (name == null) return "ERROR no primary display";
    DEVMODE cur = NewDevMode();
    if (!EnumDisplaySettingsA(name, ENUM_CURRENT_SETTINGS, ref cur)) return "ERROR cannot read primary mode";
    return "OK " + name + " " + cur.dmPelsWidth + "x" + cur.dmPelsHeight + " "
      + cur.dmDisplayFrequency + " " + cur.dmBitsPerPel + " " + cur.dmDisplayOrientation
      + " " + cur.dmDisplayFixedOutput + " " + cur.dmDisplayFlags;
  }

  public static string Apply(uint hz, bool testOnly) {
    string name = PrimaryName();
    if (name == null) return "ERROR no primary display";
    DEVMODE cur = NewDevMode();
    if (!EnumDisplaySettingsA(name, ENUM_CURRENT_SETTINGS, ref cur)) return "ERROR cannot read primary mode";
    if (cur.dmDisplayFrequency == hz) return "OK already " + hz;

    DEVMODE target = NewDevMode();
    bool found = false;
    for (int m = 0; ; m++) {
      DEVMODE mode = NewDevMode();
      if (!EnumDisplaySettingsA(name, m, ref mode)) break;
      if (mode.dmPelsWidth == cur.dmPelsWidth && mode.dmPelsHeight == cur.dmPelsHeight
          && mode.dmBitsPerPel == cur.dmBitsPerPel && mode.dmDisplayFrequency == hz
          && mode.dmDisplayOrientation == cur.dmDisplayOrientation
          && mode.dmDisplayFixedOutput == cur.dmDisplayFixedOutput
          && mode.dmDisplayFlags == cur.dmDisplayFlags) {
        target = mode;
        found = true;
        break;
      }
    }
    if (!found) return "ERROR " + hz + " Hz unavailable at " + cur.dmPelsWidth + "x" + cur.dmPelsHeight;

    int test = ChangeDisplaySettingsExA(name, ref target, IntPtr.Zero, CDS_TEST, IntPtr.Zero);
    if (test != 0) return "ERROR CDS_TEST rejected " + hz + " Hz code " + test;
    if (testOnly) return "OK testable " + hz;
    int rc = ChangeDisplaySettingsExA(name, ref target, IntPtr.Zero, 0, IntPtr.Zero);
    return rc == 0 ? "OK applied " + hz : "ERROR apply failed code " + rc;
  }
}
'@

if ($Query) {
  Write-Output ([FgDisplay]::Current())
  exit 0
}
if ($Hz -lt 24 -or $Hz -gt 1000) { Write-Output "ERROR Hz must be in [24, 1000]"; exit 1 }
$result = [FgDisplay]::Apply([uint32]$Hz, [bool]$TestOnly)
Write-Output $result
if ($result.StartsWith('ERROR')) { exit 1 }
