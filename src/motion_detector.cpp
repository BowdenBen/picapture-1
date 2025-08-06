
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <string>

#include "motion_detector.h"

using namespace cv;
using namespace std;


const vector<ColorRange> colorRanges = {
    {Scalar(100, 100, 50), Scalar(130, 255, 255)}, // Blue
    {Scalar(0, 100, 50), Scalar(10, 255, 255)},    // Red part 1
    {Scalar(160, 100, 50), Scalar(179, 255, 255)}, // Red part 2
    {Scalar(40, 70, 50), Scalar(80, 255, 255)},    // Green
    {Scalar(20, 100, 100), Scalar(30, 255, 255)},  // Yellow
    {Scalar(0, 0, 200), Scalar(179, 30, 255)}       // White
};

cv::Mat applyMorphology(const cv::Mat& src, int erodeSize, int dilateSize) {
    Mat dst = src.clone();
    if (erodeSize > 0) {
        Mat erodeKernel = getStructuringElement(MORPH_ELLIPSE, Size(erodeSize * 2 + 1, erodeSize * 2 + 1));
        erode(dst, dst, erodeKernel);
    }
    if (dilateSize > 0) {
        Mat dilateKernel = getStructuringElement(MORPH_ELLIPSE, Size(dilateSize * 2 + 1, dilateSize * 2 + 1));
        dilate(dst, dst, dilateKernel);
    }
    return dst;
}

void handleRecording(VideoWriter& writer, const Mat& frame, bool& isRecording, int& recordingFrames, int maxFrames) {
    if (isRecording) {
        writer.write(frame);
        recordingFrames++;
        if (recordingFrames >= maxFrames) {
            writer.release();
            isRecording = false;
            recordingFrames = 0;
            cout << "Stopped recording." << endl;
        }
    }
}

int main() {
    // ——— setup camera, constants, ranges, etc. ———
    string pipeline =
        "libcamerasrc ! video/x-raw, width=800, height=600 ! "
        "videoconvert ! videoscale ! video/x-raw, width=400, height=300 ! "
        "videoflip method=rotate-180 ! appsink drop=true max_buffers=2";
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Could not open camera." << endl;
        return 1;
    }

    // ———  Define recording parameters ———
    const double writeFps          = 15.0;            // playback fps
    const int    recordDurationS  = 30;               // seconds
    const int    maxRecordingFrames
        = int(writeFps * recordDurationS + 0.5);     // 450 frames

    const double motionThreshold   = 30.0;            // px
    const int    quietPeriod       = 10;              // s
    const int    erodeSize         = 2;
    const int    dilateSize        = 2;

    vector<Point> lastCentroids(colorRanges.size(), Point(-1, -1));

    while (true) {
        // ——— 1) Wait for Enter ———
        cout << "Press Enter to start one detection/recording cycle...";
        cin.get();

        // reset state for this cycle
        fill(lastCentroids.begin(), lastCentroids.end(), Point(-1, -1));
        time_t lastSnapshot = time(nullptr) - quietPeriod;
        bool  isRecording   = false;
        int   recordingFrames = 0;
        VideoWriter writer;

        // ——— 2) Inner loop — detect & record one clip ———
        while (true) {
            Mat frame, hsv;
            cap.read(frame);
            if (frame.empty()) {
                cerr << "Camera disconnected!" << endl;
                return 1;
            }

            time_t now = time(nullptr);

            if (!isRecording && difftime(now, lastSnapshot) >= quietPeriod) {
                bool movementDetected = false;

                // loop over each color
                for (size_t i = 0; i < colorRanges.size(); ++i) {
                    cvtColor(frame, hsv, COLOR_BGR2HSV);

                    Mat mask;
                    if (i == 1) {
                        Mat m1, m2;
                        inRange(hsv, colorRanges[1].lower, colorRanges[1].upper, m1);
                        inRange(hsv, colorRanges[2].lower, colorRanges[2].upper, m2);
                        bitwise_or(m1, m2, mask);
                    } else {
                        inRange(hsv, colorRanges[i].lower, colorRanges[i].upper, mask);
                    }

                    Mat proc = applyMorphology(mask, erodeSize, dilateSize);
                    Moments M = moments(proc, true);
                    if (M.m00 > 0) {
                        Point c(int(M.m10/M.m00), int(M.m01/M.m00));
                        if (lastCentroids[i] != Point(-1, -1) &&
                            norm(c - lastCentroids[i]) > motionThreshold)
                        {
                            // start recording
                            writer.open("motion.avi",
                                        VideoWriter::fourcc('M','J','P','G'),
                                        writeFps, frame.size());
                            isRecording = true;
                            recordingFrames = 0;
                            cout << "Started recording due to motion." << endl;
                            movementDetected = true;
                            break;
                        }
                        lastCentroids[i] = c;
                    }
                }

                if (!movementDetected) {
                    lastSnapshot = now;
                }
            }

            // ——— 3) Write & stop ———
            if (isRecording) {
                writer.write(frame);
                if (++recordingFrames >= maxRecordingFrames) {
                    writer.release();
                    isRecording = false;
                    cout << "Stopped recording after one clip." << endl;
                    break;  // out of inner loop
                }
            }

            if (waitKey(1) == 27) {
                cap.release();
                if (writer.isOpened()) writer.release();
                return 0;
            }
        }

        // inner loop done → go back to waiting for Enter
    }

    return 0;
}
