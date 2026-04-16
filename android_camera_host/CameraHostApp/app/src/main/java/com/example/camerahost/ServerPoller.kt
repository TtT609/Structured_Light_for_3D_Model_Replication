package com.example.camerahost

import android.util.Log
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * ServerPoller — background thread that mirrors App.tsx polling logic.
 *
 * Loop (every 500 ms):
 *   1. GET {serverUrl}/poll_command
 *   2. If action=="capture" and id is new → call onCapture() → POST PNG to /upload
 *   3. Report connection status back via onStateChange callback
 */
class ServerPoller(
    private val serverUrl: String,
    private val onCapture: () -> ByteArray,         // Called on background thread; returns PNG bytes
    private val onStateChange: (State) -> Unit,     // Always called on background thread; caller posts to UI
) {
    enum class State { DISCONNECTED, CONNECTED, CAPTURING, UPLOADING, ERROR }

    private val tag = "ServerPoller"

    @Volatile private var running = false
    private var thread: Thread? = null
    private var lastProcessedId = ""

    fun start() {
        if (running) return
        running = true
        thread = Thread({
            Log.i(tag, "Polling started → $serverUrl")
            while (running) {
                try {
                    val (action, id) = pollCommand()

                    if (action == "capture" && id.isNotEmpty() && id != lastProcessedId) {
                        lastProcessedId = id

                        onStateChange(State.CAPTURING)
                        val png = onCapture()

                        onStateChange(State.UPLOADING)
                        uploadPng(png)

                        onStateChange(State.CONNECTED)
                    } else {
                        onStateChange(State.CONNECTED)
                    }
                } catch (e: Exception) {
                    Log.w(tag, "Poll error: $e")
                    onStateChange(if (running) State.ERROR else State.DISCONNECTED)
                }

                Thread.sleep(500)
            }
            Log.i(tag, "Polling stopped.")
        }, "sl-poller")
        thread!!.isDaemon = true
        thread!!.start()
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
    }

    // -----------------------------------------------------------------------
    // Network helpers
    // -----------------------------------------------------------------------

    /**
     * GET /poll_command → returns Pair(action, id).
     * Uses a 2.5 s read timeout (server long-polls for up to 2 s).
     */
    private fun pollCommand(): Pair<String, String> {
        val conn = (URL("$serverUrl/poll_command").openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 3_000
            readTimeout = 4_000   // server may hold for up to 2 s
        }
        return try {
            val code = conn.responseCode
            if (code != 200) throw RuntimeException("poll_command returned $code")
            val body = conn.inputStream.bufferedReader().readText()
            // Parse simple JSON: {"action":"capture","id":"..."}
            val action = extractJsonString(body, "action") ?: "idle"
            val id = extractJsonString(body, "id") ?: ""
            Pair(action, id)
        } finally {
            conn.disconnect()
        }
    }

    /**
     * POST PNG bytes to /upload as multipart/form-data with field name "file"
     * and filename "capture.png" — matching what the web frontend sends.
     */
    private fun uploadPng(pngBytes: ByteArray) {
        Log.i(tag, "Uploading PNG (${pngBytes.size / 1024} KB)...")
        val boundary = "SLBoundary${System.currentTimeMillis()}"
        val conn = (URL("$serverUrl/upload").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            connectTimeout = 5_000
            readTimeout = 60_000   // large PNG may take a moment on Wi-Fi
        }
        try {
            DataOutputStream(conn.outputStream).use { out ->
                // Part header
                out.writeBytes("--$boundary\r\n")
                out.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"capture.png\"\r\n")
                out.writeBytes("Content-Type: image/png\r\n")
                out.writeBytes("\r\n")
                out.write(pngBytes)
                out.writeBytes("\r\n")
                out.writeBytes("--$boundary--\r\n")
                out.flush()
            }
            val responseCode = conn.responseCode
            if (responseCode != 200) throw RuntimeException("Upload failed: HTTP $responseCode")
            Log.i(tag, "Upload OK.")
        } finally {
            conn.disconnect()
        }
    }

    // -----------------------------------------------------------------------
    // Minimal JSON parser — avoids library dependencies
    // -----------------------------------------------------------------------

    /** Extract a string value from flat JSON: {"key":"value",...} */
    private fun extractJsonString(json: String, key: String): String? {
        val pattern = Regex("\"$key\"\\s*:\\s*\"([^\"]*)\"")
        return pattern.find(json)?.groupValues?.get(1)
    }
}
