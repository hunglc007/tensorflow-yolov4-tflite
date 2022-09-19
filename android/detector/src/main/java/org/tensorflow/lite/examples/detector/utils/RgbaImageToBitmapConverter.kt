package org.tensorflow.lite.examples.detector.utils

import android.graphics.Bitmap
import android.media.Image

// TODO: RenderScript is deprecated in API 31 and newer. Additional Vulkan implementation is recommended.
/**
 * Utility class for converting [Image] to [Bitmap].
 * Based on:
 * https://github.com/android/camera-samples/blob/master/CameraUtils/lib/src/main/java/com/example/android/camera/utils/Yuv.kt
 */
class RgbaImageToBitmapConverter : ImageToBitmapConverter {


    override fun imageToBitmap(image: Image): Bitmap {
        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(image.planes[0].buffer)

        return bitmap
    }

}