package org.tensorflow.lite.examples.detector.visualization

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View

/**
 * A simple [View] providing a render callback for [MultiBoxTracker].
 */
class TrackingOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var mTracker: MultiBoxTracker? = null


    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        mTracker?.draw(canvas)
    }

    fun setTracker(tracker: MultiBoxTracker){
        mTracker = tracker
    }
}