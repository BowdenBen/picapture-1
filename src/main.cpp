
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Color ranges for HSV
struct ColorRange {
    Scalar lower;
    Scalar upper;
};

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
    string pipeline = "libcamerasrc ! video/x-raw, width=800, height=600 ! videoconvert ! videoscale ! video/x-raw, width=400, height=300 ! videoflip method=rotate-180 ! appsink drop=true max_buffers=2";
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Could not open camera." << endl;
        return 1;
    }

    Mat frame, hsv;
    vector<Point> lastCentroids(colorRanges.size(), Point(-1, -1));
    VideoWriter writer;
    bool isRecording = false;
    int recordingFrames = 0;
    int maxRecordingFrames = 30 * 15;

    int erodeSize = 2, dilateSize = 2;
    time_t lastSnapshot = time(0) - 10;

    cout << "Press Enter to simulate alert trigger...";
    cin.get();

    for (;;) {
        cap.read(frame);
        if (frame.empty()) break;

        time_t now = time(0);
        if (!isRecording && difftime(now, lastSnapshot) >= 10) {
            bool movementDetected = false;
            for (size_t i = 0; i < colorRanges.size(); ++i) {
                cap.read(frame);
                if (frame.empty()) continue;

                cvtColor(frame, hsv, COLOR_BGR2HSV);
                Mat mask;

                if (i == 1) {
                    Mat mask1, mask2;
                    inRange(hsv, colorRanges[1].lower, colorRanges[1].upper, mask1);
                    inRange(hsv, colorRanges[2].lower, colorRanges[2].upper, mask2);
                    bitwise_or(mask1, mask2, mask);
                } else {
                    inRange(hsv, colorRanges[i].lower, colorRanges[i].upper, mask);
                }

                Mat processed = applyMorphology(mask, erodeSize, dilateSize);
                Moments m = moments(processed, true);
                if (m.m00 > 0) {
                    int cx = int(m.m10 / m.m00);
                    int cy = int(m.m01 / m.m00);
                    Point currentCentroid(cx, cy);

                    if (lastCentroids[i] != Point(-1, -1)) {
                        double dist = norm(currentCentroid - lastCentroids[i]);
                        if (dist > 30) {
                            writer.open("motion.avi", VideoWriter::fourcc('M','J','P','G'), 15, frame.size());
                            isRecording = true;
                            recordingFrames = 0;
                            cout << "Started recording due to motion." << endl;
                            movementDetected = true;
                            break;
                        }
                    }
                    lastCentroids[i] = currentCentroid;
                }
            }
            if (!movementDetected) {
                lastSnapshot = now;
            }
        }

        handleRecording(writer, frame, isRecording, recordingFrames, maxRecordingFrames);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    return 0;
}
