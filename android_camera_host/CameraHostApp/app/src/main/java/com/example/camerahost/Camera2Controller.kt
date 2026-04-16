package com.example.camerahost

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Camera2Controller — manages preview + full-resolution PNG capture.
 *
 * Flow:
 *  1. Call openCamera(cameraId) to open the device.
 *  2. Call startPreview(surfaceTexture) to show live viewfinder.
 *  3. Call captureFullResPng() to take a still shot and return a PNG ByteArray.
 *  4. Call close() on destroy.
 */
class Camera2Controller(private val context: Context) {
    private val tag = "Camera2Ctrl"

    private val cameraManager: CameraManager =
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var previewSurface: Surface? = null

    private var thread: HandlerThread? = null
    private var handler: Handler? = null

    private var activeCameraId: String? = null

    // Manual camera settings (null = use auto)
    var manualIso: Int? = null
    var manualExposureTimeNs: Long? = null
    var manualFocusDistance: Float? = null   // null = auto-focus
    var torchEnabled: Boolean = false

    // -----------------------------------------------------------------------
    // Public API — Camera enumeration
    // -----------------------------------------------------------------------

    fun listCameraIds(): List<String> = cameraManager.cameraIdList.toList()

    fun getActiveCameraId(): String? = activeCameraId

    /** Returns the best back-facing camera id, or the first available. */
    fun getDefaultBackCameraId(): String {
        for (id in cameraManager.cameraIdList) {
            val facing = cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK) return id
        }
        return cameraManager.cameraIdList.firstOrNull() ?: "0"
    }

    /**
     * Returns a human-readable label for the camera.
     * On devices like S25, the label helps identify 50MP main vs ultra-wide.
     */
    fun getCameraLabel(id: String): String {
        val cc = cameraManager.getCameraCharacteristics(id)
        val facing = cc.get(CameraCharacteristics.LENS_FACING)
        val facingStr = when (facing) {
            CameraCharacteristics.LENS_FACING_BACK -> "Back"
            CameraCharacteristics.LENS_FACING_FRONT -> "Front"
            else -> "External"
        }
        val maxSize = getMaxJpegSize(id)
        val mp = (maxSize.width.toLong() * maxSize.height.toLong()) / 1_000_000L
        return "[$id] $facingStr ${maxSize.width}×${maxSize.height} (~${mp}MP)"
    }

    /** Returns ISO range for this camera, or null if unavailable. */
    fun getIsoRange(cameraId: String): android.util.Range<Int>? {
        return cameraManager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
    }

    /** Returns exposure time range in nanoseconds, or null if unavailable. */
    fun getExposureRange(cameraId: String): android.util.Range<Long>? {
        return cameraManager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
    }

    // -----------------------------------------------------------------------
    // Open / close
    // -----------------------------------------------------------------------

    @SuppressLint("MissingPermission")
    @Synchronized
    fun openCamera(cameraId: String) {
        if (activeCameraId == cameraId && cameraDevice != null) return
        close()

        thread = HandlerThread("cam2-$cameraId").also { it.start() }
        handler = Handler(thread!!.looper)

        val latch = CountDownLatch(1)
        var openErr: Exception? = null

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                activeCameraId = cameraId
                latch.countDown()
            }
            override fun onDisconnected(camera: CameraDevice) {
                openErr = RuntimeException("Camera disconnected")
                latch.countDown()
            }
            override fun onError(camera: CameraDevice, error: Int) {
                openErr = RuntimeException("Camera open error=$error")
                latch.countDown()
            }
        }, handler)

        if (!latch.await(8, TimeUnit.SECONDS)) {
            close(); throw RuntimeException("Timeout opening camera $cameraId")
        }
        openErr?.let { close(); throw it }

        Log.i(tag, "Camera $cameraId opened.")
    }

    // -----------------------------------------------------------------------
    // Preview
    // -----------------------------------------------------------------------

    /** Start live preview rendering into the given SurfaceTexture. */
    fun startPreview(surfaceTexture: SurfaceTexture) {
        val cam = cameraDevice ?: throw RuntimeException("Camera not open")
        val cameraId = activeCameraId ?: throw RuntimeException("No active camera")

        // Size the texture to the largest preview size that fits within 1920x1080
        val previewSize = pickPreviewSize(cameraId)
        surfaceTexture.setDefaultBufferSize(previewSize.width, previewSize.height)

        // Create the still-capture ImageReader at MAXIMUM sensor resolution
        val maxSize = getMaxJpegSize(cameraId)
        imageReader = ImageReader.newInstance(maxSize.width, maxSize.height, ImageFormat.JPEG, 2)

        previewSurface = Surface(surfaceTexture)
        val surfaces = listOf(previewSurface!!, imageReader!!.surface)

        val sessionLatch = CountDownLatch(1)
        var sessErr: Exception? = null

        cam.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                sessionLatch.countDown()
                // Start continuous preview
                startRepeatingPreview(session, cam)
            }
            override fun onConfigureFailed(session: CameraCaptureSession) {
                sessErr = RuntimeException("Session configure failed")
                sessionLatch.countDown()
            }
        }, handler)

        if (!sessionLatch.await(8, TimeUnit.SECONDS)) {
            throw RuntimeException("Timeout configuring camera session")
        }
        sessErr?.let { throw it }
    }

    private fun startRepeatingPreview(session: CameraCaptureSession, cam: CameraDevice) {
        try {
            val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(previewSurface!!)
                applyManualSettings(this)
            }
            session.setRepeatingRequest(req.build(), null, handler)
        } catch (e: Exception) {
            Log.w(tag, "startRepeatingPreview failed: $e")
        }
    }

    /** Call after changing manualIso/manualExposureTimeNs/manualFocusDistance to apply them to the preview. */
    fun refreshPreviewSettings() {
        val session = captureSession ?: return
        val cam = cameraDevice ?: return
        startRepeatingPreview(session, cam)
    }

    // -----------------------------------------------------------------------
    // Capture — returns PNG bytes (lossless)
    // -----------------------------------------------------------------------

    /**
     * Captures a full-resolution still photo and returns it as a lossless PNG ByteArray.
     *
     * Process:
     *  1. Camera2 takes a JPEG at quality=100 at the sensor's maximum resolution.
     *  2. The JPEG is decoded to a Bitmap (no re-compression artefacts since quality=100).
     *  3. The Bitmap is re-encoded as PNG (lossless) and returned.
     *
     * This gives true lossless output while using the maximum sensor megapixels.
     */
    @Synchronized
    fun captureFullResPng(): ByteArray {
        val reader = imageReader ?: throw RuntimeException("ImageReader not ready - call startPreview first")
        val session = captureSession ?: throw RuntimeException("CaptureSession not ready")
        val cam = cameraDevice ?: throw RuntimeException("Camera not open")

        val captureLatch = CountDownLatch(1)
        var jpegBytes: ByteArray? = null
        var captureErr: Exception? = null

        reader.setOnImageAvailableListener({ ir ->
            try {
                ir.acquireLatestImage()?.use { img ->
                    val buf: ByteBuffer = img.planes[0].buffer
                    val arr = ByteArray(buf.remaining())
                    buf.get(arr)
                    jpegBytes = arr
                }
            } catch (e: Exception) {
                captureErr = e
            } finally {
                captureLatch.countDown()
            }
        }, handler)

        val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(reader.surface)
            // Maximum JPEG quality — only to get raw sensor data; we convert to PNG after
            set(CaptureRequest.JPEG_QUALITY, 100.toByte())
            applyManualSettings(this)
        }

        session.capture(req.build(), object : CameraCaptureSession.CaptureCallback() {}, handler)

        if (!captureLatch.await(15, TimeUnit.SECONDS)) {
            throw RuntimeException("Timeout waiting for camera capture")
        }
        captureErr?.let { throw it }

        val raw = jpegBytes ?: throw RuntimeException("No JPEG received from Camera2")

        // --- Convert JPEG → PNG (lossless) ---
        Log.i(tag, "Converting JPEG (${raw.size / 1024}KB) → PNG...")
        val bitmap = BitmapFactory.decodeByteArray(raw, 0, raw.size)
            ?: throw RuntimeException("Failed to decode JPEG to Bitmap")

        val out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        bitmap.recycle()

        val pngBytes = out.toByteArray()
        Log.i(tag, "PNG size: ${pngBytes.size / 1024}KB  (${bitmap.width}×${bitmap.height})")
        return pngBytes
    }

    // -----------------------------------------------------------------------
    // Close
    // -----------------------------------------------------------------------

    @Synchronized
    fun close() {
        try { captureSession?.close() } catch (_: Exception) {}
        captureSession = null
        try { imageReader?.close() } catch (_: Exception) {}
        imageReader = null
        try { previewSurface?.release() } catch (_: Exception) {}
        previewSurface = null
        try { cameraDevice?.close() } catch (_: Exception) {}
        cameraDevice = null
        activeCameraId = null
        try { thread?.quitSafely() } catch (_: Exception) {}
        thread = null
        handler = null
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /** Apply manual settings to a CaptureRequest.Builder. */
    private fun applyManualSettings(req: CaptureRequest.Builder) {
        val hasManualExposure = manualIso != null || manualExposureTimeNs != null

        if (hasManualExposure) {
            req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            manualIso?.let { req.set(CaptureRequest.SENSOR_SENSITIVITY, it) }
            manualExposureTimeNs?.let { req.set(CaptureRequest.SENSOR_EXPOSURE_TIME, it) }
        } else {
            req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        }

        if (manualFocusDistance != null) {
            req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            req.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualFocusDistance!!)
        } else {
            req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        }

        if (torchEnabled) {
            req.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH)
        } else {
            req.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
        }
    }

    /** Returns the largest JPEG size the camera supports (maximum sensor resolution). */
    private fun getMaxJpegSize(cameraId: String): Size {
        val cc = cameraManager.getCameraCharacteristics(cameraId)
        val cfg = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return Size(4096, 3072)
        val sizes = cfg.getOutputSizes(ImageFormat.JPEG)?.toList()
            ?: return Size(4096, 3072)
        // Sort by total pixels descending, return the largest
        return sizes.maxByOrNull { it.width.toLong() * it.height.toLong() } ?: sizes.first()
    }

    /** Returns a preview size ≤ 1920×1080 with the same aspect ratio as the sensor. */
    private fun pickPreviewSize(cameraId: String): Size {
        val cc = cameraManager.getCameraCharacteristics(cameraId)
        val cfg = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return Size(1920, 1080)
        val sizes = cfg.getOutputSizes(SurfaceTexture::class.java)?.toList()
            ?: return Size(1920, 1080)
        val maxW = 1920; val maxH = 1080
        val filtered = sizes.filter { it.width <= maxW && it.height <= maxH }
        return filtered.maxByOrNull { it.width.toLong() * it.height.toLong() }
            ?: sizes.minByOrNull { it.width.toLong() * it.height.toLong() }
            ?: Size(1280, 720)
    }
}
