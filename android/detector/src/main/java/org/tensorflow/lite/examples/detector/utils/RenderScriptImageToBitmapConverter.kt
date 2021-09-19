package org.tensorflow.lite.examples.detector.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.renderscript.*

// TODO: RenderScript is deprecated in API 31 and newer. Additional Vulkan implementation is recommended.
/**
 * Utility class for converting [Image] to [Bitmap].
 * Based on:
 * https://github.com/android/camera-samples/blob/master/CameraUtils/lib/src/main/java/com/example/android/camera/utils/Yuv.kt
 */
class RenderScriptImageToBitmapConverter(
    context: Context,
    image: Image
) : ImageToBitmapConverter {
    private val yuvBufferLength: Int = YuvByteBuffer(image).buffer.capacity()

    private val bitmap: Bitmap =
        Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)

    private val renderScript: RenderScript = RenderScript.create(context)

    private val scriptYuvToRgb: ScriptIntrinsicYuvToRGB =
        ScriptIntrinsicYuvToRGB.create(renderScript, Element.U8_3(renderScript))

    private val yuvType = Type.Builder(renderScript, Element.U8(renderScript))
        .setX(yuvBufferLength)
        .setYuvFormat(ImageFormat.YUV_420_888)
        .create()

    private val inputAllocation: Allocation =
        Allocation.createSized(
            renderScript,
            yuvType.element,
            yuvBufferLength
        )

    private val outputAllocation: Allocation = Allocation.createFromBitmap(renderScript, bitmap)

    private var bytes: ByteArray = ByteArray(yuvBufferLength)

    init {
        scriptYuvToRgb.setInput(inputAllocation)
    }

    override fun imageToBitmap(image: Image): Bitmap {
        val yuvBuffer = YuvByteBuffer(image)

        yuvBuffer.buffer.get(bytes)

        inputAllocation.copyFrom(bytes)
        scriptYuvToRgb.forEach(outputAllocation)
        outputAllocation.copyTo(bitmap)

        return bitmap
    }

}