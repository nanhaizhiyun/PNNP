package com.example.darkframecollector

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.*
import android.util.Log
import android.util.Size
import android.view.WindowManager
import android.widget.Toast
import android.app.AlertDialog
import android.graphics.Bitmap
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.max

/**
 * 采集 Activity 基类：相机生命周期、burst 采集、DNG 落盘、raw 均值质检工具。
 * 子类（暗帧/GT）只需实现自己的 UI 与采集流程。
 */
abstract class BaseCaptureActivity : Activity() {

    protected var cbThread: HandlerThread? = null
    protected var cbHandler: Handler? = null
    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null

    protected var cameraDevice: CameraDevice? = null
    protected var captureSession: CameraCaptureSession? = null
    protected var imageReader: ImageReader? = null
    protected var characteristics: CameraCharacteristics? = null
    protected var rawSize: Size? = null

    @Volatile protected var running = false
    protected var sessionDir: File? = null
    protected val frameInfos = JSONArray()

    private var pendingGranted: (() -> Unit)? = null
    private var reqCodeCounter = 100

    protected open val tag: String get() = "Capture"
    /** 输出根目录名（外存 Pictures 下） */
    protected open val rootDirName: String get() = "DarkFrameCollector"

    protected fun keepScreenOn() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    // ---------------- 权限 ----------------

    protected fun ensureCameraPermission(onGranted: () -> Unit) {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            onGranted()
        } else {
            pendingGranted = onGranted
            requestPermissions(arrayOf(Manifest.permission.CAMERA), reqCodeCounter)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>,
                                            grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            pendingGranted?.invoke()
        } else {
            toast("需要相机权限")
        }
        pendingGranted = null
    }

    // ---------------- 后台执行框架 ----------------

    /** 在后台线程执行采集流程：自动处理相机开关与 session.json 落盘。
     *  @param onDone 流程结束后回主线程回调（用于恢复 UI） */
    protected fun runInBg(onDone: (() -> Unit)? = null, block: () -> Unit) {
        bgThread = HandlerThread("camera-bg").also { it.start() }
        bgHandler = Handler(bgThread!!.looper)
        bgHandler?.post {
            try {
                openCamera()
                block()
                logUi("采集完成: ${sessionDir?.absolutePath}")
            } catch (e: Exception) {
                Log.e(tag, "采集失败", e)
                logUi("错误: ${e.message}")
            } finally {
                closeCamera()
                writeSessionJson()
                bgThread?.quitSafely()
                onDone?.let { runOnUiThread(it) }
            }
        }
    }

    // ---------------- 采集核心 ----------------

    /**
     * 以固定 (ISO, 曝光) 连续拍 count 帧 RAW，逐帧存 DNG。
     * 结果与图像按 SENSOR_TIMESTAMP 双向配对（两条回调同线程但顺序不保证）。
     *
     * @param onFrame 每帧落盘后的回调（DNG 已写入，Image 未关闭），可用于质检
     * @return 实际保存帧数
     */
    protected fun captureBurst(iso: Int, expNs: Long, expMs: Double, count: Int,
                               outDir: File, nameFn: (Int) -> String,
                               onFrame: ((img: Image, file: File, idx: Int, mean: Double) -> Unit)? = null): Int {
        val session = captureSession ?: throw IllegalStateException("会话未就绪")
        val cc = characteristics ?: throw IllegalStateException("相机未就绪")
        outDir.mkdirs()

        val results = HashMap<Long, TotalCaptureResult>()
        val pendingImgs = HashMap<Long, Image>()
        val latch = CountDownLatch(count)
        var saved = 0

        fun tryHandleFrame(ts: Long) {
            val img = pendingImgs[ts] ?: return
            val result = results[ts] ?: return
            pendingImgs.remove(ts); results.remove(ts)
            try {
                val idx = saved++
                val out = File(outDir, nameFn(idx))
                FileOutputStream(out).use { fos ->
                    DngCreator(cc, result).writeImage(fos, img)
                }
                val mean = rawMean(img)
                frameInfos.put(JSONObject()
                    .put("file", out.name)
                    .put("iso", iso)
                    .put("exposure_ms", expMs)
                    .put("mean_dn", mean)
                    .put("timestamp_ns", ts))
                onFrame?.invoke(img, out, idx, mean)
            } catch (e: Exception) {
                Log.e(tag, "存帧失败", e); logUi("存帧失败: ${e.message}")
            } finally {
                img.close()
                latch.countDown()
            }
        }

        val captureCallback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest,
                                            result: TotalCaptureResult) {
                result.get(CaptureResult.SENSOR_TIMESTAMP)?.let { results[it] = result; tryHandleFrame(it) }
            }
        }

        val reader = imageReader ?: throw IllegalStateException("ImageReader 未就绪")
        reader.setOnImageAvailableListener({ ir ->
            val img = ir.acquireNextImage() ?: return@setOnImageAvailableListener
            val ts = img.timestamp
            pendingImgs[ts] = img
            tryHandleFrame(ts)
        }, cbHandler)

        // 手动模式：锁定 ISO/曝光/黑电平
        val requests = (0 until count).map {
            session.device.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL).apply {
                addTarget(reader.surface)
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, expNs)
                if (cc.availableCaptureRequestKeys.contains(CaptureRequest.BLACK_LEVEL_LOCK)) {
                    set(CaptureRequest.BLACK_LEVEL_LOCK, true)
                }
            }.build()
        }
        session.captureBurst(requests, captureCallback, cbHandler)
        val done = latch.await(expNs / 1000000 * (count + 10).toLong() + 30000, TimeUnit.MILLISECONDS)
        if (!done) {
            logUi("警告: 超时，未配对图像 ${pendingImgs.size} 张 / 结果 ${results.size} 条")
            cbHandler?.post {
                pendingImgs.values.forEach { it.close() }
                pendingImgs.clear()
            }
        }
        reader.setOnImageAvailableListener(null, null)
        return saved
    }

    /** RAW_SENSOR 图像的抽样均值（stride 感知，每 16 像素抽 1）。*/
    protected fun rawMean(img: Image): Double {
        val plane = img.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        val w = img.width; val h = img.height
        var sum = 0.0; var n = 0L
        val row = ByteArray(rowStride)
        var y = 0
        while (y < h) {
            buf.position(y * rowStride)
            buf.get(row, 0, rowStride)
            var x = 0
            while (x < w) {
                val off = x * pixStride
                val v = (row[off].toInt() and 0xFF) or ((row[off + 1].toInt() and 0xFF) shl 8)
                sum += v; n++
                x += 16
            }
            y += 16
        }
        return if (n > 0) sum / n else 0.0
    }

    /** 黑电平估计（4 通道均值） */
    protected fun blackLevel(): Double {
        val pattern = characteristics?.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN) ?: return 0.0
        var bl = 0.0
        for (i in 0 until 4) bl += pattern.getOffsetForIndex(i % 2, i / 2)
        return bl / 4
    }

    /** 白电平（不可得时给大默认值） */
    protected fun whiteLevel(): Double {
        return (characteristics?.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 16383).toDouble()
    }

    protected fun isoRangeStr(): String =
        characteristics?.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.toString() ?: "?"
    protected fun expRangeStr(): String {
        val r = characteristics?.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE) ?: return "?"
        return "${r.lower / 1000000.0}~${r.upper / 1000000.0}ms"
    }

    // ---------------- 相机生命周期 ----------------

    protected fun openCamera() {
        if (cbThread == null) { cbThread = HandlerThread("camera-cb").also { it.start() }; cbHandler = Handler(cbThread!!.looper) }
        val cb = cbHandler!!
        val cm = getSystemService(CAMERA_SERVICE) as CameraManager
        var chosen: String? = null
        for (id in cm.cameraIdList) {
            val cc = cm.getCameraCharacteristics(id)
            val facing = cc.get(CameraCharacteristics.LENS_FACING)
            val caps = cc.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: continue
            val hasRaw = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
            val hasManual = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
            if (facing == CameraCharacteristics.LENS_FACING_BACK && hasRaw && hasManual) { chosen = id; break }
        }
        if (chosen == null) throw IllegalStateException("该设备不支持 RAW 或手动传感器控制")
        characteristics = cm.getCameraCharacteristics(chosen)

        val streamCfgs = characteristics!!.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = streamCfgs?.getOutputSizes(ImageFormat.RAW_SENSOR)
            ?: throw IllegalStateException("RAW_SENSOR 不可用")
        rawSize = sizes.maxByOrNull { it.width * it.height }
        logUi("相机 $chosen, RAW 尺寸 $rawSize")

        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        sessionDir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "$rootDirName/$ts")
        sessionDir!!.mkdirs()

        imageReader = ImageReader.newInstance(rawSize!!.width, rawSize!!.height,
            ImageFormat.RAW_SENSOR, /*maxImages=*/ 16)

        val latch = CountDownLatch(1)
        val openErr = arrayOfNulls<String>(1)
        cm.openCamera(chosen, object : CameraDevice.StateCallback() {
            override fun onOpened(cd: CameraDevice) { cameraDevice = cd; latch.countDown() }
            override fun onDisconnected(cd: CameraDevice) { cd.close(); openErr[0] = "相机被断开"; latch.countDown() }
            override fun onError(cd: CameraDevice, error: Int) {
                cd.close()
                openErr[0] = when (error) {
                    CameraDevice.StateCallback.ERROR_CAMERA_IN_USE -> "相机正被其他应用占用 (ERROR_CAMERA_IN_USE)"
                    CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE -> "打开相机数已达上限 (ERROR_MAX_CAMERAS_IN_USE)"
                    CameraDevice.StateCallback.ERROR_CAMERA_DISABLED -> "相机被设备策略禁用 (ERROR_CAMERA_DISABLED)"
                    CameraDevice.StateCallback.ERROR_CAMERA_DEVICE -> "相机设备致命错误 (ERROR_CAMERA_DEVICE)"
                    CameraDevice.StateCallback.ERROR_CAMERA_SERVICE -> "相机服务致命错误 (ERROR_CAMERA_SERVICE)"
                    else -> "未知错误码 $error"
                }
                latch.countDown()
            }
        }, cb)
        if (!latch.await(5, TimeUnit.SECONDS)) openErr[0] = "打开相机超时（5 秒无回调）"

        val dev = cameraDevice ?: throw IllegalStateException("相机打开失败: ${openErr[0] ?: "未知原因（检查相机权限是否已授予）"}")
        val sLatch = CountDownLatch(1)
        dev.createCaptureSession(listOf(imageReader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) { captureSession = s; sLatch.countDown() }
            override fun onConfigureFailed(s: CameraCaptureSession) { sLatch.countDown() }
        }, cb)
        if (!sLatch.await(5, TimeUnit.SECONDS) || captureSession == null)
            throw IllegalStateException("会话创建失败")
    }

    protected fun closeCamera() {
        captureSession?.close(); captureSession = null
        cameraDevice?.close(); cameraDevice = null
        imageReader?.close(); imageReader = null
        cbThread?.quitSafely(); cbThread = null; cbHandler = null
    }

    /** 子类可附加的 session 元数据（GT 模式、场景名等） */
    protected open fun sessionExtras(): JSONObject = JSONObject()

    protected fun writeSessionJson() {
        val dir = sessionDir ?: return
        val cc = characteristics ?: return
        val jo = JSONObject()
            .put("device", Build.MANUFACTURER + " " + Build.MODEL)
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("iso_range", cc.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.toString())
            .put("exposure_range_ns", cc.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.toString())
            .put("white_level", cc.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL))
            .put("black_level", blackLevel())
            .put("raw_size", rawSize?.toString())
        val extra = sessionExtras()
        for (k in extra.keys()) jo.put(k, extra.get(k))
        jo.put("frames", frameInfos)
        File(dir, "session.json").writeText(jo.toString(2))
        logUi("元数据: ${File(dir, "session.json").absolutePath}")
    }

    // ---------------- 抓拍测试 ----------------

    /** 以界面指定的 ISO/曝光手动抓拍一帧 RAW（与正式采集一致的锁定项），评估亮度并弹窗显示缩略图 */
    protected fun captureTestShot(iso: Int?, expMs: Double?) {
        if (iso == null || iso <= 0 || expMs == null || expMs <= 0.0) {
            toast("请先在界面填写有效的 ISO 和曝光时间，再抓拍测试")
            return
        }
        ensureCameraPermission {
            logUi("抓拍测试中…")
            runInBg {
                val session = captureSession ?: throw IllegalStateException("会话未就绪")
                val reader = imageReader ?: throw IllegalStateException("ImageReader 未就绪")
                val latch = CountDownLatch(1)
                var shot: Image? = null
                reader.setOnImageAvailableListener({ ir ->
                    val im = ir.acquireNextImage()
                    if (im != null) { shot?.close(); shot = im; latch.countDown() }
                }, cbHandler)
                // 手动模式：与正式采集完全一致的锁定项（AE/AF 关闭、黑电平锁定）
                val cc = characteristics ?: throw IllegalStateException("相机未就绪")
                // 拍前校验：ISO/曝光超出设备能力范围直接告警（硬件会静默钳位）
                val reqNs = (expMs * 1000000).toLong()
                var preWarn = ""
                val isoR = cc.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                if (isoR != null && (iso < isoR.lower || iso > isoR.upper))
                    preWarn += "\n⚠ ISO$iso 超出设备范围 ${isoR.lower}~${isoR.upper}，将被钳位"
                val expR = cc.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                if (expR != null && (reqNs < expR.lower || reqNs > expR.upper))
                    preWarn += "\n⚠ 曝光 ${expMs}ms 超出设备范围 ${"%.1f".format(expR.lower / 1000000.0)}~${"%.1f".format(expR.upper / 1000000.0)}ms，将被钳位"
                if (preWarn.isNotEmpty()) logUi("参数告警:$preWarn")
                val req = session.device.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL).apply {
                    addTarget(reader.surface)
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                    set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                    set(CaptureRequest.SENSOR_EXPOSURE_TIME, (expMs * 1000000).toLong())
                    if (cc.availableCaptureRequestKeys.contains(CaptureRequest.BLACK_LEVEL_LOCK)) {
                        set(CaptureRequest.BLACK_LEVEL_LOCK, true)
                    }
                }.build()
                logUi("抓拍: 手动 ISO$iso ${expMs}ms")
                var shotResult: TotalCaptureResult? = null
                val capCb = object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest,
                                                    result: TotalCaptureResult) { shotResult = result }
                }
                session.capture(req, capCb, cbHandler)
                if (!latch.await(8, TimeUnit.SECONDS)) {
                    reader.setOnImageAvailableListener(null, null)
                    throw IllegalStateException("抓拍超时")
                }
                // 结果回调通常紧随图像，稍等片刻取 ISO/曝光
                var waitMs = 0
                while (shotResult == null && waitMs < 1000) { Thread.sleep(50); waitMs += 50 }
                reader.setOnImageAvailableListener(null, null)
                val img = shot ?: throw IllegalStateException("未取得图像")
                try {
                    val (mean, satFrac) = rawStats(img, 4)
                    val bmp = rawToBitmap(img, 640)
                    val actIso = shotResult?.get(CaptureResult.SENSOR_SENSITIVITY)
                    val actExpNs = shotResult?.get(CaptureResult.SENSOR_EXPOSURE_TIME)
                    var note = preWarn
                    if (actIso != null && actIso != iso && !note.contains("ISO$iso "))
                        note += "\n⚠ 请求 ISO$iso 超出设备范围，实际生效 ISO$actIso（范围 ${isoRangeStr()}）"
                    if (actExpNs != null && kotlin.math.abs(actExpNs - reqNs) > reqNs / 20 && !note.contains("曝光 ${expMs}ms "))
                        note += "\n⚠ 请求曝光 ${expMs}ms 超出设备范围，实际生效 ${"%.1f".format(actExpNs / 1000000.0)}ms（范围 ${expRangeStr()}）"
                    showTestShot(bmp, mean, satFrac, actIso, actExpNs, note)
                } finally {
                    img.close()
                }
            }
        }
    }

    /** 抽样统计：均值 DN 与接近白电平的饱和像素比例（step 为抽样间隔） */
    protected fun rawStats(img: Image, step: Int): Pair<Double, Double> {
        val plane = img.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        val w = img.width; val h = img.height
        val satTh = whiteLevel() * 0.98
        var sum = 0.0; var n = 0L; var sat = 0L
        val row = ByteArray(rowStride)
        var y = 0
        while (y < h) {
            buf.position(y * rowStride)
            buf.get(row, 0, rowStride)
            var x = 0
            while (x < w) {
                val off = x * pixStride
                val v = (row[off].toInt() and 0xFF) or ((row[off + 1].toInt() and 0xFF) shl 8)
                sum += v; n++
                if (v >= satTh) sat++
                x += step
            }
            y += step
        }
        return (if (n > 0) sum / n else 0.0) to (if (n > 0) sat.toDouble() / n else 0.0)
    }

    /** RAW 16-bit 灰度缩略图（按黑/白电平归一化，长边约 maxDim 像素） */
    protected fun rawToBitmap(img: Image, maxDim: Int): Bitmap {
        val plane = img.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixStride = plane.pixelStride
        val w = img.width; val h = img.height
        val step = max(1, max(w, h) / maxDim)
        val bw = (w + step - 1) / step; val bh = (h + step - 1) / step
        // 先采样原始值，再按 99.5% 分位数拉伸显示（传感器白电平远大于实际曝光，直接按白电平归一会全黑）
        val vals = IntArray(bw * bh)
        val row = ByteArray(rowStride)
        var y = 0; var j = 0
        while (y < h) {
            buf.position(y * rowStride)
            buf.get(row, 0, rowStride)
            var x = 0; var i = 0
            while (x < w && i < bw) {
                val off = x * pixStride
                vals[j * bw + i] = (row[off].toInt() and 0xFF) or ((row[off + 1].toInt() and 0xFF) shl 8)
                x += step; i++
            }
            y += step; j++
        }
        val sorted = vals.sortedArray()
        val hi = sorted[(sorted.size * 0.995).toInt().coerceIn(0, sorted.size - 1)]
        val lo = minOf(blackLevel().toInt(), hi - 1)
        val scale = 255.0 / max(1.0, (hi - lo).toDouble())
        val pixels = IntArray(bw * bh)
        for (k in vals.indices) {
            var g = ((vals[k] - lo) * scale).toInt()
            if (g < 0) g = 0 else if (g > 255) g = 255
            pixels[k] = (0xFF shl 24) or (g shl 16) or (g shl 8) or g
        }
        return Bitmap.createBitmap(pixels, bw, bh, Bitmap.Config.ARGB_8888)
    }

    /** 弹窗显示缩略图与亮度结论 */
    private fun showTestShot(bmp: Bitmap, mean: Double, satFrac: Double,
                             iso: Int?, expNs: Long?, note: String = "") {
        val black = blackLevel(); val white = whiteLevel()
        val dyn = white - black
        val verdict = when {
            satFrac > 0.01 -> "过曝：${"%.1f".format(satFrac * 100)}% 像素接近白电平"
            mean < black + dyn * 0.05 -> "太暗：均值不足动态范围 5%"
            else -> "亮度正常"
        }
        val expo = "ISO ${iso?.toString() ?: "?"}，曝光 ${expNs?.let { "%.1f".format(it / 1000000.0) } ?: "?"} ms"
        val detail = "$expo\n均值 ${"%.0f".format(mean)} DN（黑电平 ${"%.0f".format(black)}，白电平 ${"%.0f".format(white)}），饱和像素 ${"%.2f".format(satFrac * 100)}%"
        logUi("抓拍: $verdict；$detail")
        runOnUiThread {
            val tv = TextView(this).apply { text = "$verdict\n$detail$note"; setPadding(40, 24, 40, 8) }
            val iv = ImageView(this).apply { setImageBitmap(bmp); adjustViewBounds = true }
            val box = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                addView(tv); addView(iv)
            }
            AlertDialog.Builder(this).setTitle("抓拍测试").setView(box)
                .setPositiveButton("关闭", null).show()
        }
    }
    // ---------------- UI 工具 ----------------

    protected fun logUi(msg: String) {
        Log.i(tag, msg)
        runOnUiThread { onLogLine(msg) }
    }

    /** 子类把日志追加到自己的 TextView */
    protected abstract fun onLogLine(msg: String)

    protected fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    protected fun parseInts(s: String) = s.split(',', '，', ' ').mapNotNull { it.trim().toIntOrNull() }
    protected fun parseDoubles(s: String) = s.split(',', '，', ' ').mapNotNull { it.trim().toDoubleOrNull() }
}
