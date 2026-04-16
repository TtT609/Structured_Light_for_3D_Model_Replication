# SL Camera — Android Native Camera for Structured Light

This is a full-replacement for the web frontend browser camera.  
It uses the Android **Camera2 API** to capture at the phone's **native full sensor resolution** (e.g. 50 MP on Samsung Galaxy S25) and uploads lossless **PNG** files directly to the PC server.

---

## How It Works (Same Protocol as Web Frontend)

```
Phone                               PC  (server/server.py)
────                               ──
Poll GET /poll_command ──────────→
                       ←────────── {"action":"idle"}   (repeat every 500ms)

PC starts a scan:
                       ←────────── {"action":"capture","id":"..."}

Phone triggers Camera2 still capture at full resolution
JPEG @ quality-100 → decode → PNG (lossless)
POST /upload ────────────────────→ image saved to scan folder
                       ←────────── 200 OK
```

**The server (`server/server.py`) does not need any changes.**  
The same scan / calibration workflow works with this app.

---

## Setup

### On the PC

1. Start the server as usual: `python server/main.py`
2. In the GUI → Tab **"1. Scan & Generate"** → select **"Android Native App"** radio button
3. Note the URL shown (e.g. `http://192.168.1.50:5000`)

### On the Phone

1. Build & install **CameraHostApp** via Android Studio:
   - Open `android_camera_host/CameraHostApp` in Android Studio
   - **Build → Run** (or Build APK and install manually)
2. Open the **SL Camera** app
3. Tap **≡ Settings** → type the PC URL → tap **Save & Close**
4. Status badge turns 🟢 **LINKED** when connected

---

## Features

| Feature | Details |
|---|---|
| **Full resolution** | Captures at the camera sensor's maximum JPEG size (e.g. 12000×9000 on S25 = 108 MP) |
| **Lossless PNG** | JPEG @quality=100 → PNG conversion on device before upload — no compression artefacts |
| **Live viewfinder** | Full-screen Camera2 preview |
| **Camera selector** | Choose between main/wide/tele cameras in Settings |
| **Pro Mode** | Manual ISO, shutter speed, focus distance via Camera2 sliders |
| **Log overlay** | Last 5 status messages shown on screen |
| **Status badge** | DISCONNECTED / LINKED / CAPTURING / UPLOADING |

---

## Wi-Fi Setup (Recommended)

Ensure phone and PC are on the same Wi-Fi network.  
Use the IP shown in the GUI (e.g. `http://192.168.1.50:5000`).

## USB Setup (Optional — Lower Latency)

```bash
adb devices
adb reverse tcp:5000 tcp:5000
```
Then use `http://127.0.0.1:5000` as the server URL in the app.

---

## Important: Redo Calibration After First Switch

After switching from the web frontend to this Android app, you **must redo calibration once**, because the camera intrinsic parameters (focal length, principal point) at 50 MP differ from those at 8 MP.

**Use the same PC workflow:**
1. GUI Tab 1 → "Capture Calib Images" (app captures via Camera2, uploads to PC)
2. GUI Tab 1 → "Compute Calibration"

This generates a new `calib.mat` valid for the full-resolution sensor.
