# Run on happycomputer in PowerShell AS ADMINISTRATOR.
# Exposes WSL backend (uvicorn on port 12212) to LAN + Tailscale.
#
# Prerequisites:
#   - Backend running in WSL: python -m uvicorn server:app --host 0.0.0.0 --port 12212
#   - Tailscale running on Windows (happycomputer online in admin console)

$PORT = 12212
$LISTEN = "0.0.0.0"

$wslRaw = (wsl hostname -I 2>$null)
if (-not $wslRaw) {
    Write-Error "Could not get WSL IP. Is WSL running?"
    exit 1
}
$WSL_IP = ($wslRaw.Trim() -split "\s+")[0]
Write-Host "WSL IP: $WSL_IP"

netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=$LISTEN 2>$null
netsh interface portproxy add v4tov4 listenaddress=$LISTEN listenport=$PORT connectaddress=$WSL_IP connectport=$PORT

$ruleName = "VOS Annotation Backend TCP $PORT"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if (-not $existing) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $PORT | Out-Null
    Write-Host "Firewall rule added: $ruleName"
} else {
    Write-Host "Firewall rule already exists: $ruleName"
}

Write-Host ""
Write-Host "Port proxy (Windows -> WSL):"
netsh interface portproxy show all
Write-Host ""
Write-Host "Test on Windows:"
Write-Host "  curl http://127.0.0.1:$PORT/health"
Write-Host "  curl http://100.66.192.124:$PORT/health"
Write-Host ""
Write-Host "Test on Mac (Tailscale):"
Write-Host "  curl http://100.66.192.124:$PORT/health"
Write-Host ""
Write-Host "Note: WSL IP can change after reboot — re-run this script if connection breaks."
