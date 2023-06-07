package org.tensorflow.lite.examples.detector.ui.detector

import android.view.View
import android.widget.ImageView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import org.tensorflow.lite.examples.detector.R

class CameraBottomSheetCallback(
    private val arrowImageView: ImageView
) : BottomSheetBehavior.BottomSheetCallback() {

    override fun onStateChanged(bottomSheet: View, newState: Int) {
        when (newState) {
            BottomSheetBehavior.STATE_HIDDEN -> {
            }
            BottomSheetBehavior.STATE_EXPANDED -> {
                arrowImageView.setImageResource(R.drawable.icn_chevron_down)
            }
            BottomSheetBehavior.STATE_COLLAPSED -> {
                arrowImageView.setImageResource(R.drawable.icn_chevron_up)
            }
            BottomSheetBehavior.STATE_DRAGGING -> {
            }
            BottomSheetBehavior.STATE_SETTLING -> {
                arrowImageView.setImageResource(R.drawable.icn_chevron_up)
            }
            BottomSheetBehavior.STATE_HALF_EXPANDED -> {
            }
        }
    }

    override fun onSlide(bottomSheet: View, slideOffset: Float) {}

}