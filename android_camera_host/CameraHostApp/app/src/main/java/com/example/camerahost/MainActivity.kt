package com.example.camerahost

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.TextureView
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity(), TextureView.SurfaceTextureListener {

    private val tag = "MainActivity"
    private val mainHandler = Handler(Looper.getMainLooper())

    // ── Views ────────────────────────────────────────────────────────────────
    private lateinit var textureView: TextureView
    private lateinit var topBar: View
    private lateinit var txtStatus: TextView
    private lateinit var btnProMode: Button
    private lateinit var btnSettings: Button
    private lateinit var logContainer: LinearLayout
    private lateinit var imgThumbnail: ImageView
    private lateinit var uploadingOverlay: FrameLayout
    private lateinit var btnShutter: FrameLayout
    private lateinit var settingsPanel: LinearLayout
    private lateinit var proModePanel: LinearLayout

    // Settings panel
    private lateinit var editServerUrl: EditText
    private lateinit var spinnerCamera: Spinner
    private lateinit var btnSaveSettings: Button

    // Pro mode panel
    private lateinit var seekIso: SeekBar
    private lateinit var txtIso: TextView
    private lateinit var seekExpTime: SeekBar
    private lateinit var txtExpTime: TextView
    private lateinit var seekFocus: SeekBar
    private lateinit var txtFocus: TextView
    private lateinit var chkManualFocus: CheckBox
    private lateinit var chkManualExposure: CheckBox

    // ── Camera / Poller ──────────────────────────────────────────────────────
    private var camera: Camera2Controller? = null
    private var poller: ServerPoller? = null

    // ISO / exposure step tables (log-scale for better usability)
    private val isoSteps = listOf(50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)
    private val expSteps = listOf(                    // nanoseconds
        1_000_000L,   // 1ms  = 1/1000 s
        2_000_000L,   // 1/500
        4_000_000L,   // 1/250
        8_000_000L,   // 1/125
        16_000_000L,  // 1/60
        33_000_000L,  // 1/30
        66_000_000L,  // 1/15
        133_000_000L, // 1/8
        250_000_000L, // 1/4
        500_000_000L, // 1/2
        1_000_000_000L, // 1s
    )
    private val focusSteps = 101   // 0 = auto, 1-100 = manual (0.0 to 1.0 diopters)

    // ── State ─────────────────────────────────────────────────────────────────
    private var serverUrl: String = ""
    private var proModeVisible = false
    private var settingsVisible = false
    private val logMessages = ArrayDeque<String>(5)

    // ── Prefs key ─────────────────────────────────────────────────────────────
    private val prefsName = "sl_prefs"
    private val prefUrl = "server_url"
    private val prefCamera = "camera_id"

    // =========================================================================
    // Lifecycle
    // =========================================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()
        setupListeners()
        loadPrefs()
    }

    override fun onResume() {
        super.onResume()
        if (hasCameraPermission()) {
            initCamera()
        } else {
            requestCameraPermission()
        }
    }

    override fun onPause() {
        super.onPause()
        poller?.stop()
        poller = null
        camera?.close()
        camera = null
    }

    // =========================================================================
    // View binding
    // =========================================================================

    private fun bindViews() {
        textureView      = findViewById(R.id.textureView)
        topBar           = findViewById(R.id.topBar)
        txtStatus        = findViewById(R.id.txtStatus)
        btnProMode       = findViewById(R.id.btnProMode)
        btnSettings      = findViewById(R.id.btnSettings)
        logContainer     = findViewById(R.id.logContainer)
        imgThumbnail     = findViewById(R.id.imgThumbnail)
        uploadingOverlay = findViewById(R.id.uploadingOverlay)
        btnShutter       = findViewById(R.id.btnShutter)
        settingsPanel    = findViewById(R.id.settingsPanel)
        proModePanel     = findViewById(R.id.proModePanel)

        editServerUrl    = findViewById(R.id.editServerUrl)
        spinnerCamera    = findViewById(R.id.spinnerCamera)
        btnSaveSettings  = findViewById(R.id.btnSaveSettings)

        seekIso          = findViewById(R.id.seekIso)
        txtIso           = findViewById(R.id.txtIso)
        seekExpTime      = findViewById(R.id.seekExpTime)
        txtExpTime       = findViewById(R.id.txtExpTime)
        seekFocus        = findViewById(R.id.seekFocus)
        txtFocus         = findViewById(R.id.txtFocus)
        chkManualFocus   = findViewById(R.id.chkManualFocus)
        chkManualExposure= findViewById(R.id.chkManualExposure)
    }

    // =========================================================================
    // Button listeners
    // =========================================================================

    private fun setupListeners() {

        // Shutter button → manual capture
        btnShutter.setOnClickListener {
            val cam = camera ?: return@setOnClickListener
            if (poller == null) return@setOnClickListener   // not connected
            Thread {
                try {
                    onUiThread { setStatus(ServerPoller.State.CAPTURING) }
                    val png = cam.captureFullResPng()
                    onUiThread { setStatus(ServerPoller.State.UPLOADING) }
                    uploadManual(png)
                    onUiThread {
                        addLog("Manual capture uploaded.")
                        setStatus(ServerPoller.State.CONNECTED)
                    }
                } catch (e: Exception) {
                    onUiThread {
                        addLog("Capture error: ${e.message}")
                        setStatus(ServerPoller.State.ERROR)
                    }
                }
            }.also { it.isDaemon = true }.start()
        }

        // Pro Mode toggle
        btnProMode.setOnClickListener {
            proModeVisible = !proModeVisible
            settingsVisible = false
            settingsPanel.visibility = View.GONE
            proModePanel.visibility = if (proModeVisible) View.VISIBLE else View.GONE
            btnProMode.setBackgroundColor(
                if (proModeVisible) 0x883B82F4.toInt() else 0x44FFFFFF
            )
        }

        // Settings toggle
        btnSettings.setOnClickListener {
            settingsVisible = !settingsVisible
            proModeVisible = false
            proModePanel.visibility = View.GONE
            settingsPanel.visibility = if (settingsVisible) View.VISIBLE else View.GONE
        }

        // Save settings
        btnSaveSettings.setOnClickListener {
            val url = editServerUrl.text.toString().trim().trimEnd('/')
            val selectedCam = spinnerCamera.selectedItem as? String ?: ""
            if (url.isNotEmpty()) {
                serverUrl = url
                savePrefs(url, selectedCam)
                settingsPanel.visibility = View.GONE
                settingsVisible = false
                restartPolling()
            }
        }

        // Seek bar listeners for Pro Mode
        setupSeekBarListeners()
    }

    private fun setupSeekBarListeners() {
        seekIso.max = isoSteps.size  // 0 = auto, 1..n = manual
        seekExpTime.max = expSteps.size
        seekFocus.max = focusSteps

        seekIso.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, byUser: Boolean) {
                val cam = camera ?: return
                if (!chkManualExposure.isChecked) { txtIso.text = "AUTO"; return }
                if (p == 0) {
                    cam.manualIso = null
                    txtIso.text = "AUTO"
                } else {
                    val iso = isoSteps.getOrElse(p - 1) { isoSteps.last() }
                    cam.manualIso = iso
                    txtIso.text = iso.toString()
                }
                cam.refreshPreviewSettings()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        seekExpTime.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, byUser: Boolean) {
                val cam = camera ?: return
                if (!chkManualExposure.isChecked) { txtExpTime.text = "AUTO"; return }
                if (p == 0) {
                    cam.manualExposureTimeNs = null
                    txtExpTime.text = "AUTO"
                } else {
                    val ns = expSteps.getOrElse(p - 1) { expSteps.last() }
                    cam.manualExposureTimeNs = ns
                    txtExpTime.text = formatShutter(ns)
                }
                cam.refreshPreviewSettings()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        seekFocus.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, byUser: Boolean) {
                val cam = camera ?: return
                if (!chkManualFocus.isChecked) { txtFocus.text = "AUTO"; return }
                if (p == 0) {
                    cam.manualFocusDistance = null
                    txtFocus.text = "AUTO"
                } else {
                    val dist = (p.toFloat() / focusSteps.toFloat())
                    cam.manualFocusDistance = dist
                    txtFocus.text = "%.2f".format(dist)
                }
                cam.refreshPreviewSettings()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        chkManualFocus.setOnCheckedChangeListener { _, checked ->
            val cam = camera ?: return@setOnCheckedChangeListener
            if (!checked) { cam.manualFocusDistance = null; txtFocus.text = "AUTO" }
            cam.refreshPreviewSettings()
        }

        chkManualExposure.setOnCheckedChangeListener { _, checked ->
            val cam = camera ?: return@setOnCheckedChangeListener
            if (!checked) {
                cam.manualIso = null; cam.manualExposureTimeNs = null
                txtIso.text = "AUTO"; txtExpTime.text = "AUTO"
            }
            cam.refreshPreviewSettings()
        }
    }

    // =========================================================================
    // Camera initialisation
    // =========================================================================

    private fun initCamera() {
        val savedCamId = getPrefs().getString(prefCamera, null)
        camera = Camera2Controller(this).also { ctrl ->
            // Populate camera spinner
            val ids = ctrl.listCameraIds()
            val labels = ids.map { ctrl.getCameraLabel(it) }
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels)
                .also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            spinnerCamera.adapter = adapter
            val defaultId = savedCamId ?: ctrl.getDefaultBackCameraId()
            val idx = ids.indexOf(defaultId).takeIf { it >= 0 } ?: 0
            spinnerCamera.setSelection(idx)

            // Open camera
            Thread {
                try {
                    ctrl.openCamera(ids[idx])
                    // If TextureView is already ready, start preview immediately
                    val st = textureView.surfaceTexture
                    if (textureView.isAvailable && st != null) {
                        ctrl.startPreview(st)
                        addLog("Camera ready: ${ctrl.getCameraLabel(ids[idx])}")
                    }
                    // Start polling
                    if (serverUrl.isNotEmpty()) startPolling()
                } catch (e: Exception) {
                    Log.e(tag, "Camera open failed: $e")
                    onUiThread { addLog("Camera error: ${e.message}") }
                }
            }.also { it.isDaemon = true }.start()
        }
        textureView.surfaceTextureListener = this
    }

    // =========================================================================
    // TextureView.SurfaceTextureListener
    // =========================================================================

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, w: Int, h: Int) {
        val cam = camera ?: return
        Thread {
            try { cam.startPreview(surface) } catch (e: Exception) {
                Log.e(tag, "startPreview failed: $e")
            }
        }.also { it.isDaemon = true }.start()
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, w: Int, h: Int) {}
    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}

    // =========================================================================
    // Polling
    // =========================================================================

    private fun startPolling() {
        if (serverUrl.isEmpty()) return
        poller?.stop()
        addLog("Connecting to $serverUrl…")
        poller = ServerPoller(
            serverUrl = serverUrl,
            onCapture  = {
                camera?.captureFullResPng() ?: throw RuntimeException("Camera not ready")
            },
            onStateChange = { state ->
                onUiThread {
                    setStatus(state)
                    if (state == ServerPoller.State.CONNECTED) addLog("Linked to PC.")
                    if (state == ServerPoller.State.UPLOADING) {
                        addLog("Captured & uploading PNG…")
                        refreshThumbnail()
                    }
                }
            }
        )
        poller!!.start()
    }

    private fun restartPolling() {
        poller?.stop()
        poller = null
        startPolling()
    }

    /** Manual shutter upload — reuse the same /upload endpoint. */
    private fun uploadManual(png: ByteArray) {
        val url = "$serverUrl/upload"
        val boundary = "SLBoundary${System.currentTimeMillis()}"
        val conn = (java.net.URL(url).openConnection() as java.net.HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            connectTimeout = 5_000
            readTimeout = 60_000
        }
        java.io.DataOutputStream(conn.outputStream).use { out ->
            out.writeBytes("--$boundary\r\n")
            out.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"capture.png\"\r\n")
            out.writeBytes("Content-Type: image/png\r\n\r\n")
            out.write(png)
            out.writeBytes("\r\n--$boundary--\r\n")
        }
        conn.responseCode   // trigger send
        conn.disconnect()
    }

    // =========================================================================
    // UI state helpers
    // =========================================================================

    private fun setStatus(state: ServerPoller.State) {
        val (label, color, bgColor) = when (state) {
            ServerPoller.State.CONNECTED    -> Triple("LINKED",       0xFF4ADE80.toInt(), 0x33166534.toInt())
            ServerPoller.State.CAPTURING    -> Triple("CAPTURING",    0xFFFBBF24.toInt(), 0x33713F12.toInt())
            ServerPoller.State.UPLOADING    -> Triple("UPLOADING",    0xFFFBBF24.toInt(), 0x33713F12.toInt())
            ServerPoller.State.ERROR        -> Triple("ERROR",        0xFFEF4444.toInt(), 0x337F1D1D.toInt())
            ServerPoller.State.DISCONNECTED -> Triple("DISCONNECTED", 0xFFEF4444.toInt(), 0x337F1D1D.toInt())
        }
        txtStatus.text = label
        txtStatus.setTextColor(color)
        txtStatus.setBackgroundColor(bgColor)

        val uploading = state == ServerPoller.State.UPLOADING || state == ServerPoller.State.CAPTURING
        uploadingOverlay.visibility = if (uploading) View.VISIBLE else View.GONE
        btnShutter.isEnabled = !uploading
        btnShutter.alpha = if (uploading) 0.5f else 1.0f
    }

    private fun addLog(msg: String) {
        val full = "[${java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())}] $msg"
        logMessages.addFirst(full)
        while (logMessages.size > 5) logMessages.removeLast()
        refreshLogContainer()
    }

    private fun refreshLogContainer() {
        logContainer.removeAllViews()
        for (log in logMessages) {
            val tv = TextView(this).apply {
                text = log
                setTextColor(0xFF4ADE80.toInt())
                textSize = 9f
                setBackgroundColor(0x80000000.toInt())
                setPadding(8, 3, 8, 3)
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
                lp.setMargins(0, 2, 0, 2)
                layoutParams = lp
            }
            logContainer.addView(tv)
        }
    }

    private fun refreshThumbnail() {
        // Show placeholder immediately; a proper thumbnail would need extra capture
        imgThumbnail.visibility = View.VISIBLE
    }

    private fun formatShutter(ns: Long): String {
        return when {
            ns >= 1_000_000_000L -> "%.1fs".format(ns / 1_000_000_000.0)
            else -> "1/${(1_000_000_000L / ns).toInt()}"
        }
    }

    // =========================================================================
    // Permissions
    // =========================================================================

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() =
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == 1001 && results.isNotEmpty() && results[0] == PackageManager.PERMISSION_GRANTED) {
            initCamera()
        }
    }

    // =========================================================================
    // SharedPreferences
    // =========================================================================

    private fun getPrefs() = getSharedPreferences(prefsName, Context.MODE_PRIVATE)

    private fun loadPrefs() {
        val prefs = getPrefs()
        serverUrl = prefs.getString(prefUrl, "") ?: ""
        editServerUrl.setText(serverUrl)
    }

    private fun savePrefs(url: String, cameraId: String) {
        getPrefs().edit().putString(prefUrl, url).putString(prefCamera, cameraId).apply()
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private fun onUiThread(action: () -> Unit) = mainHandler.post(action)
}
