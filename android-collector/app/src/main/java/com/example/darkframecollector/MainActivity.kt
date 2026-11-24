package com.example.darkframecollector

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.widget.Button

/** 入口：选择采集任务 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<Button>(R.id.btnGt).setOnClickListener {
            startActivity(Intent(this, GtActivity::class.java))
        }
        findViewById<Button>(R.id.btnDark).setOnClickListener {
            startActivity(Intent(this, DarkActivity::class.java))
        }
        findViewById<Button>(R.id.btnFlat).setOnClickListener {
            startActivity(Intent(this, FlatActivity::class.java))
        }
        findViewById<Button>(R.id.btnCheck).setOnClickListener {
            val report = checkReprocessCapability(
                getSystemService(CAMERA_SERVICE) as CameraManager)
            AlertDialog.Builder(this)
                .setTitle("reprocess 能力自检")
                .setMessage(report)
                .setPositiveButton("OK", null)
                .show()
        }
    }

    /**
     * 检查设备 reprocess 能力（决定去噪后的 raw 能否塞回原厂 ISP）。
     * 读取相机特性不需要相机权限。
     */
    private fun checkReprocessCapability(cm: CameraManager): String {
        val report = StringBuilder()
        for (id in cm.cameraIdList) {
            val cc = cm.getCameraCharacteristics(id)
            val caps = cc.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()

            // 注意：Android 没有 RAW_REPROCESS 常量；raw 可塞回 = RAW 能力位
            // + reprocess 输入格式包含 RAW_SENSOR
            val hasRawCapability = caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
            var hasYuvReprocess = caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_YUV_REPROCESSING)
            val hasPrivReprocess = caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING)

            var inputFmts: IntArray? = null
            if (Build.VERSION.SDK_INT >= 26) {
                inputFmts = cc.get(CameraCharacteristics.SCALER_AVAILABLE_INPUT_FORMATS)
            }
            val hasRawReprocess = hasRawCapability &&
                inputFmts != null && inputFmts.contains(ImageFormat.RAW_SENSOR)
            if (inputFmts != null && inputFmts.contains(ImageFormat.YUV_420_888)) {
                // 部分设备输入格式列出了 YUV 但未声明 YUV_REPROCESS 标志，以格式为准
                hasYuvReprocess = true
            }

            report.append("相机 ").append(id).append(":\n")
            report.append("  RAW_REPROCESS=").append(hasRawReprocess)
                .append("  YUV_REPROCESS=").append(hasYuvReprocess)
                .append("  PRIVATE_REPROCESS=").append(hasPrivReprocess).append('\n')

            report.append("  输入格式: ").append(inputFmts?.joinToString {
                when (it) {
                    ImageFormat.YUV_420_888 -> "YUV_420_888"
                    ImageFormat.RAW_SENSOR -> "RAW_SENSOR"
                    ImageFormat.PRIVATE -> "PRIVATE"
                    else -> "0x" + it.toString(16)
                }
            }).append('\n')

            val level = cc.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            report.append("  硬件等级: ").append(when (level) {
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY(不支持 reprocess)"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
                else -> level.toString()
            }).append('\n')
        }

        report.append('\n').append("结论: ")
        val body = report.toString()
        report.append(when {
            body.contains("RAW_REPROCESS=true") ->
                "支持 RAW 塞回 -> 路线一(最优)：clean raw 直接作 reprocess 输入"
            body.contains("YUV_REPROCESS=true") ->
                "支持 YUV 塞回 -> 路线一：clean raw 先自 demosaic 成 YUV 再塞回"
            body.contains("PRIVATE_REPROCESS=true") ->
                "仅私有格式 -> 需查厂商文档确认格式值，或走路线二"
            else ->
                "不支持 reprocess -> 路线二：App 内 mini-ISP 自渲染闭环"
        })
        report.append("\n\n注意：声明支持仍需跑一次完整 reprocess 冒烟验证（部分设备声明与实际不符）。")
        return report.toString()
    }
}
