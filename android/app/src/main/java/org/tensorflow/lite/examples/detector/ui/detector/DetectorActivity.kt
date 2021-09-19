package org.tensorflow.lite.examples.detector.ui.detector

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.examples.detector.databinding.ActivityCameraBinding
import org.tensorflow.lite.examples.detector.extensions.getViewModelFactory
import org.tensorflow.lite.examples.detector.misc.Constants

class DetectorActivity : AppCompatActivity() {
    private companion object {
        const val CAMERA_REQUEST_CODE: Int = 1

        const val CAMERA_ASPECT_RATIO: Int = AspectRatio.RATIO_16_9
    }

    private val viewModel by viewModels<DetectorViewModel> { getViewModelFactory() }

    private lateinit var binding: ActivityCameraBinding

    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setUpBottomSheet()

        @SuppressLint("SetTextI18n")
        binding.bottomSheet.cropInfo.text =
            "${Constants.DETECTION_MODEL.inputSize}x${Constants.DETECTION_MODEL.inputSize}"

        cameraProviderFuture = ProcessCameraProvider.getInstance(baseContext)
        requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)

        viewModel.setUpDetectionProcessor(
            assets,
            resources.displayMetrics,
            binding.tovCamera,
            binding.pvCamera
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            CAMERA_REQUEST_CODE -> {
                val indexOfCameraPermission = permissions.indexOf(Manifest.permission.CAMERA)
                if (grantResults[indexOfCameraPermission] == PackageManager.PERMISSION_GRANTED) {
                    cameraProviderFuture.addListener(
                        this::bindPreview,
                        ContextCompat.getMainExecutor(baseContext)
                    )
                } else {
                    Toast.makeText(
                        baseContext,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT
                    ).show()
                    finish()
                }
            }
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun setUpBottomSheet() {
        val sheetBehavior = BottomSheetBehavior.from(binding.bottomSheet.root)
        sheetBehavior.isHideable = false

        val callback = CameraBottomSheetCallback(binding.bottomSheet.bottomSheetArrow)
        sheetBehavior.addBottomSheetCallback(callback)

        val gestureLayout = binding.bottomSheet.gestureLayout
        gestureLayout.viewTreeObserver.addOnGlobalLayoutListener {
            val height: Int = gestureLayout.measuredHeight
            sheetBehavior.peekHeight = height
        }
    }

    private fun bindPreview() {
        val preview: Preview = Preview.Builder()
            .setTargetAspectRatio(CAMERA_ASPECT_RATIO)
            .setTargetRotation(DetectorViewModel.CAMERA_ROTATION)
            .build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(binding.pvCamera.surfaceProvider)

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(CAMERA_ASPECT_RATIO)
            .setTargetRotation(DetectorViewModel.CAMERA_ROTATION)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalysis.setAnalyzer(
            ContextCompat.getMainExecutor(baseContext),
            this::analyzeImage
        )

        val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
        cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            imageAnalysis,
            preview
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun analyzeImage(image: ImageProxy) {
        @SuppressLint("SetTextI18n")
        binding.bottomSheet.frameInfo.text = "${image.width}x${image.height}"

        lifecycleScope.launch(Dispatchers.Default) {
            image.use {
                if (!viewModel.imageConvertedIsSetUpped()) {
                    viewModel.setUpImageConverter(baseContext, image)
                }

                val detectionTime = viewModel.detectObjectsOnImage(image)

                withContext(Dispatchers.Main) {
                    @SuppressLint("SetTextI18n")
                    binding.bottomSheet.timeInfo.text = "$detectionTime ms"
                }
            }
        }
    }
}
