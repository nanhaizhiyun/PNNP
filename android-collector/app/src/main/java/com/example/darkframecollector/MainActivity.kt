package com.example.darkframecollector

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
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

            // 注意：Android 没有 RAW_REPROCESS 能力常量；平台也不单独暴露
            // "raw 可塞回"的开关。判定依据 = RAW 能力位 + 硬件等级（FULL/LEVEL_3
            // 才可能支持 reprocess），最终必须以冒烟测试为准。
            val hasRawCapability = caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
            val hasYuvReprocess = caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_YUV_REPROCESSING)
            val hasPrivReprocess = caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING)

            val level = cc.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            val levelName = when (level) {
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
                else -> level.toString()
            }
            val highLevel = level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL ||
                level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3

            report.append("相机 ").append(id).append(":\n")
            report.append("  RAW 能力=").append(hasRawCapability)
                .append("  YUV_REPROCESSING=").append(hasYuvReprocess)
                .append("  PRIVATE_REPROCESSING=").append(hasPrivReprocess)
                .append("  硬件等级=").append(levelName).append('\n')
            if (hasRawCapability && highLevel) {
                report.append("  -> 可能支持 raw reprocess（需冒烟验证）\n")
            } else if (hasYuvReprocess) {
                report.append("  -> 支持 YUV reprocess（clean raw 需先 demosaic 成 YUV）\n")
            } else if (!hasPrivReprocess) {
                report.append("  -> 未见 reprocess 能力，建议走 mini-ISP 自渲染\n")
            }
        }

        report.append('\n').append(
            "说明：\n" +
            "  · 可能支持 raw reprocess  -> 路线一(最优)：clean raw 直接作 reprocess 输入\n" +
            "  · 支持 YUV reprocess     -> 路线一：clean raw 先 demosaic 成 YUV 再塞回\n" +
            "  · 仅私有格式/能力缺失    -> 路线二：App 内 mini-ISP 自渲染闭环\n" +
            "  · 声明支持不等于可用，仍需跑一次完整 reprocess 冒烟验证")
        return report.toString()
    }
}
