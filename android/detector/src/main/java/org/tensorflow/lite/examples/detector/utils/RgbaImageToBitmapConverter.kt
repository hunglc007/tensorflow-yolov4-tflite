package org.tensorflow.lite.examples.detector.utils

import android.graphics.Bitmap
import android.media.Image

/**
 * Utility class for converting [Image] to [Bitmap].
 */
class RgbaImageToBitmapConverter : ImageToBitmapConverter {
    override fun imageToBitmap(image: Image): Bitmap {
        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(image.planes[0].buffer)

        return bitmap
    }

}