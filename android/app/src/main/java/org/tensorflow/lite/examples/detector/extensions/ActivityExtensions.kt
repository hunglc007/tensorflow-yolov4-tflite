package org.tensorflow.lite.examples.detector.extensions

import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.detector.misc.ViewModelFactory

fun AppCompatActivity.getViewModelFactory(): ViewModelFactory {
    return ViewModelFactory(this)
}
