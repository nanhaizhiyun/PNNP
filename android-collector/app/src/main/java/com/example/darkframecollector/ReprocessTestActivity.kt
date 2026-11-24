package com.example.darkframecollector

import android.app.Activity
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.hardware.camera2.params.InputConfiguration
import android.media.Image
import android.media.ImageReader
import android.media.ImageWriter
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * reprocess 冒烟测试：验证"把 raw 塞回原厂 ISP"的完整链路。
 *
 * 流程（恒等测试，暂不接去噪模型）：
 *   1. 普通会话抓一帧 RAW_SENSOR + TotalCaptureResult
 *   2. 建可重处理会话（输入 RAW_SENSOR，输出 JPEG）
 *   3. 把原样 raw 通过 ImageWriter 写入 session.inputSurface
 *   4. 发 TEMPLATE_REPROCESS 请求 → ISP 出 JPEG
 *   5. 保存 JPEG + 日志，肉眼应与普通拍照一致
 *
 * 去噪模型接入点见 "TODO(去噪)"：在拷贝 raw 时对像素做处理即可。
 */
class ReprocessTestActivity : Activity() {

    private lateinit var tvLog: TextView
    private lateinit var btnStart: Button

    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null
    private var device: CameraDevice? = null
    private var rawSize: Size? = null
    private var characteristics: CameraCharacteristics? = null
    private var cameraId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_reprocess)

        tvLog = findViewById(R.id.tvLog)
        btnStart = findViewById(R.id.btnStart)
        btnStart.setOnClickListener {
            btnStart.isEnabled = false
            tvLog.text = ""
            bgThread = HandlerThread("reprocess-bg").also { it.start() }
            bgHandler = Handler(bgThread!!.looper)
            bgHandler?.post {
                try {
                    runSmokeTest()
                } catch (e: Exception) {
                    Log.e(TAG, "冒烟测试失败", e)
                    log("错误: ${e.message}")
                } finally {
                    closeAll()
                    runOnUiThread { btnStart.isEnabled = true }
                    bgThread?.quitSafely()
                }
            }
        }
    }

    companion object {
        private const val TAG = "ReprocessTest"
    }

    private fun log(msg: String) {
        Log.i(TAG, msg)
        runOnUiThread { tvLog.append("$msg\n") }
    }

    // ---------------- 主流程 ----------------

    private fun runSmokeTest() {
        val cm = getSystemService(CAMERA_SERVICE) as CameraManager
        pickCamera(cm)
        openCamera(cm)
        val dev = device ?: throw IllegalStateException("相机打开失败")
        val cc = characteristics ?: throw IllegalStateException("无相机参数")
        val size = rawSize ?: throw IllegalStateException("无 RAW 尺寸")
        log("相机 $cameraId  RAW ${size.width}x${size.height}")

        // ---- 1) 普通会话抓一帧 raw ----
        val rawReader = ImageReader.newInstance(size.width, size.height, ImageFormat.RAW_SENSOR, 2)
        val rawLatch = CountDownLatch(1)
        var capturedImage: Image? = null
        var capturedResult: TotalCaptureResult? = null
        rawReader.setOnImageAvailableListener({ r ->
            val img = r.acquireNextImage()
            if (img != null && capturedImage == null) capturedImage = img
            rawLatch.countDown()
        }, bgHandler)

        val plainSession = createSession(dev, listOf(rawReader.surface))
        val req1 = dev.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(rawReader.surface)
        }.build()
        plainSession.capture(req1, object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest,
                                            result: TotalCaptureResult) {
                if (capturedResult == null) capturedResult = result
            }
        }, bgHandler)
        if (!rawLatch.await(8, TimeUnit.SECONDS)) throw IllegalStateException("抓 raw 超时")
        val srcImage = capturedImage ?: throw IllegalStateException("未拿到 raw 帧")
        val result = capturedResult ?: throw IllegalStateException("未拿到 TotalCaptureResult")
        log("① 抓到 raw: ${srcImage.width}x${srcImage.height} format=${srcImage.format} " +
            "planes=${srcImage.planes.size}")
        plainSession.close()
        rawReader.setOnImageAvailableListener(null, null)

        // ---- 2) JPEG 输出尺寸 ----
        val jpegSize = pickJpegSize(cc, size)
        log("② JPEG 输出尺寸 ${jpegSize.width}x${jpegSize.height}")
        val jpegReader = ImageReader.newInstance(jpegSize.width, jpegSize.height, ImageFormat.JPEG, 2)
        val jpegLatch = CountDownLatch(1)
        var jpegBytes: ByteArray? = null
        jpegReader.setOnImageAvailableListener({ r ->
            val img = r.acquireNextImage()
            if (img != null) {
                val buf = img.planes[0].buffer
                jpegBytes = ByteArray(buf.remaining()).also { buf.get(it) }
                img.close()
            }
            jpegLatch.countDown()
        }, bgHandler)

        // ---- 3) 可重处理会话 ----
        val inputCfg = InputConfiguration(size.width, size.height, ImageFormat.RAW_SENSOR)
        val repSession = createReprocessableSession(dev, inputCfg, listOf(jpegReader.surface))
        // getInputSurface() 标注为可空：非可重处理会话时返回 null
        val inSurface = repSession.inputSurface
            ?: throw IllegalStateException("inputSurface 为 null（会话不是可重处理会话）")
        log("③ 可重处理会话已建立, inputSurface 可用")

        // ---- 4) 把 raw 写入 inputSurface ----
        val writer = ImageWriter.newInstance(inSurface, 1)
        val outImg = writer.dequeueInputImage()
            ?: throw IllegalStateException("dequeueInputImage 返回 null")
        if (outImg.width != size.width || outImg.height != size.height) {
            throw IllegalStateException("inputImage 尺寸不符: ${outImg.width}x${outImg.height}")
        }
        copyRawPlanes(srcImage, outImg)
        // TODO(去噪): 在这里读取 outImg.planes[0].buffer，做去噪/校正后再写回，
        //             即可把 clean raw 交给原厂 ISP。当前为恒等拷贝。
        writer.queueInputImage(outImg)
        log("④ raw 已写入 inputSurface（恒等拷贝，未去噪）")

        // ---- 5) TEMPLATE_REPROCESS 请求 ----
        val req2 = dev.createCaptureRequest(CameraDevice.TEMPLATE_REPROCESS).apply {
            addTarget(jpegReader.surface)
        }.build()
        val repLatch = CountDownLatch(1)
        repSession.capture(req2, object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest,
                                            res: TotalCaptureResult) {
                log("   重处理请求完成")
            }
        }, bgHandler)
        if (!jpegLatch.await(10, TimeUnit.SECONDS)) throw IllegalStateException("重处理输出超时")
        val bytes = jpegBytes ?: throw IllegalStateException("未拿到 JPEG 输出")

        // ---- 6) 保存 ----
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "ReprocessTest")
        dir.mkdirs()
        val out = File(dir, "reprocess_$ts.jpg")
        FileOutputStream(out).use { it.write(bytes) }
        log("⑤ 成功！JPEG ${bytes.size / 1024} KB -> ${out.absolutePath}")
        log("   -> 打开该图片应与普通拍照效果一致；说明 raw 塞回 ISP 链路可用")
        log("   -> 下一步：在 TODO(去噪) 处接入模型，输出 clean raw")
        srcImage.close()
        writer.close()
        jpegReader.close()
    }

    // ---------------- 工具 ----------------

    private fun pickCamera(cm: CameraManager) {
        for (id in cm.cameraIdList) {
            val cc = cm.getCameraCharacteristics(id)
            val caps = cc.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: continue
            val facing = cc.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK &&
                caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW) &&
                caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING)) {
                cameraId = id
                characteristics = cc
                val map = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                rawSize = map?.getOutputSizes(ImageFormat.RAW_SENSOR)?.maxByOrNull { it.width * it.height }
                return
            }
        }
        throw IllegalStateException("未找到支持 RAW + reprocess 的后置相机")
    }

    private fun pickJpegSize(cc: CameraCharacteristics, raw: Size): Size {
        val map = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: throw IllegalStateException("无 StreamConfigurationMap")
        val sizes = map.getOutputSizes(ImageFormat.JPEG) ?: throw IllegalStateException("无 JPEG 尺寸")
        // 优先完全一致，其次同宽高比且面积最接近
        sizes.firstOrNull { it.width == raw.width && it.height == raw.height }?.let { return it }
        val ar = raw.width.toFloat() / raw.height
        return sizes.filter { Math.abs(it.width.toFloat() / it.height - ar) < 0.02f }
            .minByOrNull { Math.abs(it.width * it.height - raw.width * raw.height) } ?: sizes[0]
    }

    private fun openCamera(cm: CameraManager) {
        val latch = CountDownLatch(1)
        var err: String? = null
        cm.openCamera(cameraId!!, object : CameraDevice.StateCallback() {
            override fun onOpened(cd: CameraDevice) { device = cd; latch.countDown() }
            override fun onDisconnected(cd: CameraDevice) { cd.close(); err = "相机断开"; latch.countDown() }
            override fun onError(cd: CameraDevice, e: Int) { cd.close(); err = "打开错误 $e"; latch.countDown() }
        }, bgHandler)
        if (!latch.await(5, TimeUnit.SECONDS)) err = "打开超时"
        if (device == null) throw IllegalStateException(err ?: "打开失败")
    }

    private fun createSession(dev: CameraDevice, outputs: List<android.view.Surface>): CameraCaptureSession {
        val latch = CountDownLatch(1)
        var session: CameraCaptureSession? = null
        dev.createCaptureSession(outputs, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) { session = s; latch.countDown() }
            override fun onConfigureFailed(s: CameraCaptureSession) { latch.countDown() }
        }, bgHandler)
        if (!latch.await(5, TimeUnit.SECONDS) || session == null)
            throw IllegalStateException("普通会话创建失败")
        return session!!
    }

    private fun createReprocessableSession(dev: CameraDevice, input: InputConfiguration,
                                           outputs: List<android.view.Surface>): CameraCaptureSession {
        val latch = CountDownLatch(1)
        var session: CameraCaptureSession? = null
        var fail: String? = null
        dev.createReprocessableCaptureSession(input, outputs,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) { session = s; latch.countDown() }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    fail = "onConfigureFailed"; latch.countDown()
                }
            }, bgHandler)
        if (!latch.await(5, TimeUnit.SECONDS) || session == null)
            throw IllegalStateException("可重处理会话创建失败${fail?.let { "($it)" } ?: "（超时）"}" +
                "：设备可能实际不支持 RAW 输入")
        return session!!
    }

    /** 按行拷贝 raw 像素（兼容 rowStride 差异） */
    private fun copyRawPlanes(src: Image, dst: Image) {
        val sp = src.planes[0]
        val dp = dst.planes[0]
        val bytesPerRow = minOf(src.width * 2, sp.rowStride, dp.rowStride)
        val sb = sp.buffer
        val db = dp.buffer
        val s = sb.duplicate()
        val d = db.duplicate()
        for (y in 0 until minOf(src.height, dst.height)) {
            s.position(y * sp.rowStride)
            s.limit(y * sp.rowStride + bytesPerRow)
            d.position(y * dp.rowStride)
            d.limit(y * dp.rowStride + bytesPerRow)
            d.put(s)
        }
        log("   拷贝 ${src.height} 行, 每行 $bytesPerRow 字节 (srcStride=${sp.rowStride} dstStride=${dp.rowStride})")
    }

    private fun closeAll() {
        device?.close(); device = null
    }
}
