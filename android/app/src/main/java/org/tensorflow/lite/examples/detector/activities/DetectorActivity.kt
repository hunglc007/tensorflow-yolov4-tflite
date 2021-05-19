package org.tensorflow.lite.examples.detector.activities

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.examples.detector.Detector
import org.tensorflow.lite.examples.detector.DetectorFactory
import org.tensorflow.lite.examples.detector.R
import org.tensorflow.lite.examples.detector.databinding.ActivityCameraBinding
import org.tensorflow.lite.examples.detector.enums.DetectionModel
import org.tensorflow.lite.examples.detector.helpers.DetectionProcessor
import org.tensorflow.lite.examples.detector.utils.ImageToBitmapConverter
import kotlin.system.measureTimeMillis

class DetectorActivity : AppCompatActivity() {
    private companion object {
        const val TAG: String = "DetectorActivity"

        const val CAMERA_REQUEST_CODE: Int = 1

        const val CAMERA_ASPECT_RATIO: Int = AspectRatio.RATIO_16_9

        /*
        * Use Surface.ROTATION_0 for portrait and Surface.ROTATION_270 for landscape
        */
        const val CAMERA_ROTATION: Int = Surface.ROTATION_0

        val DETECTION_MODEL: DetectionModel = MainActivity.DETECTION_MODEL
        const val MINIMUM_SCORE: Float = MainActivity.MINIMUM_SCORE
    }

    private lateinit var mBinding: ActivityCameraBinding

    private lateinit var mDetector: Detector

    private lateinit var mCameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private var mDetectionProcessor: DetectionProcessor? = null

    private var mImageConverter: ImageToBitmapConverter? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(mBinding.root)

        setUpBottomSheet()

        @SuppressLint("SetTextI18n")
        mBinding.bottomSheet.cropInfo.text =
            "${DETECTION_MODEL.inputSize}x${DETECTION_MODEL.inputSize}"

        mCameraProviderFuture = ProcessCameraProvider.getInstance(baseContext)
        requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)

        mDetector = DetectorFactory.createDetector(assets, DETECTION_MODEL, MINIMUM_SCORE)

        lifecycleScope.launch(Dispatchers.Main) {
            mDetectionProcessor = DetectionProcessor(
                context = baseContext,
                detector = mDetector,
                trackingOverlay = mBinding.tovCamera,
            )

            while (mBinding.pvCamera.childCount == 0) {
                delay(200)
            }

            val surfaceView: View = mBinding.pvCamera.getChildAt(0)
            val previewWidth: Int = surfaceView.width
            val previewHeight: Int = surfaceView.height

            mDetectionProcessor!!.initializeTrackingLayout(
                previewWidth,
                previewHeight,
                mDetector.getDetectionModel().inputSize,
                CAMERA_ROTATION
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == CAMERA_REQUEST_CODE) {
            val indexOfCameraPermission = permissions.indexOf(Manifest.permission.CAMERA)
            if (grantResults[indexOfCameraPermission] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
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

    private fun setUpBottomSheet() {
        val sheetBehavior = BottomSheetBehavior.from(mBinding.bottomSheet.root)
        sheetBehavior.isHideable = false
        sheetBehavior.addBottomSheetCallback(
            object : BottomSheetCallback() {
                override fun onStateChanged(bottomSheet: View, newState: Int) {
                    val arrowImageView = mBinding.bottomSheet.bottomSheetArrow
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
            })

        val gestureLayout = mBinding.bottomSheet.gestureLayout
        gestureLayout.viewTreeObserver.addOnGlobalLayoutListener {
            val height: Int = gestureLayout.measuredHeight
            sheetBehavior.peekHeight = height
        }
    }

    private fun startCamera() {
        mCameraProviderFuture.addListener(
            ::bindPreview,
            ContextCompat.getMainExecutor(baseContext)
        )
    }

    private fun bindPreview() {
        val preview: Preview = Preview.Builder()
            .setTargetAspectRatio(CAMERA_ASPECT_RATIO)
            .setTargetRotation(CAMERA_ROTATION)
            .build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(mBinding.pvCamera.surfaceProvider)

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(CAMERA_ASPECT_RATIO)
            .setTargetRotation(CAMERA_ROTATION)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalysis.setAnalyzer(
            ContextCompat.getMainExecutor(baseContext),
            this::analyzeImage
        )

        val cameraProvider: ProcessCameraProvider = mCameraProviderFuture.get()
        cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            imageAnalysis,
            preview
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun analyzeImage(image: ImageProxy) = lifecycleScope.launch(Dispatchers.Default) {
        image.use {
            val bitmap: Bitmap
            val conversionTime = measureTimeMillis {
                if (mImageConverter == null) {
                    Log.v(TAG, "Image size : ${image.width}x${image.height}")
                    mImageConverter = ImageToBitmapConverter(baseContext, image.image!!)
                }

                withContext(Dispatchers.Main) {
                    @SuppressLint("SetTextI18n")
                    mBinding.bottomSheet.frameInfo.text = "${image.width}x${image.height}"
                }
                bitmap = mImageConverter!!.imageToBitmap(image.image!!)
            }
            Log.v(TAG, "Conversion time : $conversionTime ms")

            val detectionTime: Long = mDetectionProcessor!!.processImage(bitmap)
            Log.v(TAG, "Detection time : $detectionTime ms")
            val processingTime = conversionTime + detectionTime
            Log.v(TAG, "Analysis time : $processingTime ms")

            withContext(Dispatchers.Main) {
                @SuppressLint("SetTextI18n")
                mBinding.bottomSheet.timeInfo.text = "$detectionTime ms"
            }
        }
    }
}
