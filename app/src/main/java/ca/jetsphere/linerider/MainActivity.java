package ca.jetsphere.linerider;

import android.Manifest;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2
{
    private static String TAG = "MainActivity";

    JavaCameraView javaCameraView;
    Mat mRgba, cropped, gray, blurred, threshold, mask, erodeElement, dilateElement, hierarchy;
    Rect roi;

    /**
     *
     */
    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS : { javaCameraView.enableView(); Log.i("OpenCV", "OpenCV loaded successfully");} break;
                default : super.onManagerConnected(status);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        setContentView(R.layout.activity_main);

        tryStartCamera();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCv loaded successfully");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.i(TAG, "OpenCv failed to load");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        gray = new Mat(height, width, CvType.CV_8UC1);
        blurred = new Mat(height, width, CvType.CV_8UC1);
        threshold = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        roi = new Rect(60, 0, 160, 120);
        cropped = new Mat(mRgba, roi);
        Imgproc.cvtColor(cropped, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, blurred, new Size(5,5), 0);
        Imgproc.threshold(blurred, threshold, 60, 255, Imgproc.THRESH_BINARY_INV);

        erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8));

        Imgproc.erode(threshold, threshold, erodeElement);
        Imgproc.dilate(threshold, threshold, dilateElement);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mask, contours, hierarchy, 1, Imgproc.CHAIN_APPROX_NONE);

        if (contours.size() == 0) return cropped;

        MatOfPoint max = null; double area = 0;
        for (MatOfPoint matOfPoint : contours)
        {
            double contourArea = Imgproc.contourArea(matOfPoint);
            max = contourArea > area ? matOfPoint : max;
        }

        Moments moment = Imgproc.moments(max);

        int x = ((int) (moment.m10/moment.m00));
        int y = ((int) (moment.m01/moment.m00));
        Imgproc.line(cropped, new Point(x,y), new Point(x, 0), new Scalar(0, 255, 0), 2);
        Imgproc.drawContours(cropped, contours, -1, new Scalar(0, 255, 0), 1);

        return cropped;
    }

    /**
     *
     */
    private void tryStartCamera() {
        javaCameraView = findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        javaCameraView.setMaxFrameSize(640, 480);
    }
}
