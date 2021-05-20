package org.tensorflow.lite.examples.detector.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.renderscript.*
import java.nio.ByteBuffer

/**
 * Utility class for converting [Image] to [Bitmap].
 */
class ImageToBitmapConverter(context: Context, image: Image) {
    private val bitmap: Bitmap =
        Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)

    private val renderScript: RenderScript = RenderScript.create(context)

    private val scriptYuvToRgb: ScriptIntrinsicYuvToRGB =
        ScriptIntrinsicYuvToRGB.create(renderScript, Element.U8_3(renderScript))

    private val elemType = Type.Builder(renderScript, Element.YUV(renderScript))
        .setYuvFormat(ImageFormat.YUV_420_888)
        .create()

    private val inputAllocation: Allocation =
        Allocation.createSized(
            renderScript,
            elemType.element,
            image.planes.sumOf { it.buffer.remaining() }
        )

    private val outputAllocation: Allocation = Allocation.createFromBitmap(renderScript, bitmap)

    init {
        scriptYuvToRgb.setInput(inputAllocation)
    }

    /**
     * Converts [Image] to [Bitmap] using [ScriptIntrinsicYuvToRGB].
     */
    fun imageToBitmap(image: Image): Bitmap {
        val yuvBuffer: ByteArray = yuv420ToByteArray(image)

        inputAllocation.copyFrom(yuvBuffer)
        scriptYuvToRgb.forEach(outputAllocation)
        outputAllocation.copyTo(bitmap)

        return bitmap
    }

    private fun yuv420ToByteArray(image: Image): ByteArray {
        val yBuffer = image.planes[0].buffer.toByteArray()
        val uBuffer = image.planes[1].buffer.toByteArray()
        val vBuffer = image.planes[2].buffer.toByteArray()

        return yBuffer + uBuffer + vBuffer
    }

    private fun ByteBuffer.toByteArray(): ByteArray {
        rewind()
        val data = ByteArray(remaining())
        get(data)
        return data
    }
}