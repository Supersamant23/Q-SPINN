# install_cuda_torch.ps1
# Resilient CUDA PyTorch installer with automatic retry on connection drop.
# Run from the Q-SPINN project root:
#   .\.venv\Scripts\activate
#   .\install_cuda_torch.ps1

$url = "https://download-r2.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp313-cp313-win_amd64.whl"
$dest = "$env:TEMP\torch_cu126.whl"
$maxRetries = 10
$attempt = 0

Write-Host "Downloading PyTorch 2.14.0+cu126 with automatic resume on disconnect..."

while ($attempt -lt $maxRetries) {
    $attempt++
    Write-Host "`n[Attempt $attempt / $maxRetries]"
    
    try {
        # Use BITS (Background Intelligent Transfer Service) — built into Windows,
        # handles resume natively without needing wget or curl
        Import-Module BitsTransfer
        Start-BitsTransfer -Source $url -Destination $dest -Description "PyTorch CUDA" -DisplayName "torch_cu126"
        Write-Host "Download complete."
        break
    } catch {
        Write-Host "Connection dropped: $($_.Exception.Message)"
        Write-Host "Retrying in 5 seconds..."
        Start-Sleep -Seconds 5
    }
}

if (Test-Path $dest) {
    Write-Host "`nInstalling wheel..."
    .venv\Scripts\pip install $dest
    Write-Host "`nVerifying..."
    .venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
} else {
    Write-Host "Download failed after $maxRetries attempts."
}
