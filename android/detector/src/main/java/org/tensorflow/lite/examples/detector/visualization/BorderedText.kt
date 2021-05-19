package org.tensorflow.lite.examples.detector.visualization

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

/**
 * A class that encapsulates the tedious bits of rendering legible, bordered text onto a [Canvas].
 */
class BorderedText(
    interiorColor: Int = Color.WHITE,
    exteriorColor: Int = Color.BLACK,
    textSize: Float = 16.0f
) {
    private val mInteriorPaint: Paint = Paint().also {
        it.textSize = textSize
        it.color = interiorColor
        it.style = Paint.Style.FILL
        it.isAntiAlias = false
        it.alpha = 255
    }

    private val mExteriorPaint: Paint = Paint().also {
        it.textSize = textSize
        it.color = exteriorColor
        it.style = Paint.Style.FILL_AND_STROKE
        it.strokeWidth = textSize / 8
        it.isAntiAlias = false
        it.alpha = 255
    }


    fun drawText(
        canvas: Canvas,
        posX: Float,
        posY: Float,
        text: String,
        bgPaint: Paint
    ) {
        val width = mExteriorPaint.measureText(text)
        val textSize = mExteriorPaint.textSize
        val paint = Paint(bgPaint)

        paint.style = Paint.Style.FILL
        paint.alpha = 160
        canvas.drawRect(posX, posY + textSize.toInt(), posX + width.toInt(), posY, paint)
        canvas.drawText(text, posX, posY + textSize, mInteriorPaint)
    }
}