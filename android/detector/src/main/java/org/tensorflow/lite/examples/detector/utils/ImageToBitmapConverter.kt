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
    private val mBitmap: Bitmap =
        Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)

    private val mRenderScript: RenderScript = RenderScript.create(context)

    private val mScriptYuvToRgb: ScriptIntrinsicYuvToRGB =
        ScriptIntrinsicYuvToRGB.create(mRenderScript, Element.U8_3(mRenderScript))

    private val mElemType = Type.Builder(mRenderScript, Element.YUV(mRenderScript))
        .setYuvFormat(ImageFormat.YUV_420_888)
        .create()

    private val mInputAllocation: Allocation =
        Allocation.createSized(
            mRenderScript,
            mElemType.element,
            image.planes.sumOf { it.buffer.remaining() }
        )

    private val mOutputAllocation: Allocation = Allocation.createFromBitmap(mRenderScript, mBitmap)

    init {
        mScriptYuvToRgb.setInput(mInputAllocation)
    }

    /**
     * Converts [Image] to [Bitmap] using [ScriptIntrinsicYuvToRGB].
     */
    fun imageToBitmap(image: Image): Bitmap {
        val yuvBuffer: ByteArray = yuv420ToByteArray(image)

        mInputAllocation.copyFrom(yuvBuffer)
        mScriptYuvToRgb.forEach(mOutputAllocation)
        mOutputAllocation.copyTo(mBitmap)

        return mBitmap
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