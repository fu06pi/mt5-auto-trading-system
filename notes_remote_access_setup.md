# Remote Access Setup

## Status
- Tailscale active
- GNOME Remote Desktop running
- RDP enabled
- VNC enabled

## Host
- Tailscale IP: 100.78.16.11
- Magic DNS: fedora.tailab1af6.ts.net

## Commands used
```bash
gsettings set org.gnome.desktop.remote-desktop.rdp enable true
gsettings set org.gnome.desktop.remote-desktop.rdp view-only false
gsettings set org.gnome.desktop.remote-desktop.rdp screen-share-mode extend

gsettings set org.gnome.desktop.remote-desktop.vnc enable true
gsettings set org.gnome.desktop.remote-desktop.vnc view-only false
gsettings set org.gnome.desktop.remote-desktop.vnc screen-share-mode extend

systemctl --user restart gnome-remote-desktop.service
```


## Disabled
- RDP: disabled
- VNC: disabled
