import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class VehicleSpeedEstimator {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String videoPath = "path/to/traffic_video.mp4";
        String cfgPath = "path/to/yolo/yolov3.cfg";
        String weightsPath = "path/to/yolo/yolov3.weights";
        String namesPath = "path/to/yolo/coco.names";

        // Load YOLO model
        Net net = Dnn.readNetFromDarknet(cfgPath, weightsPath);

        List<String> classes = loadClasses(namesPath);
        VideoCapture cap = new VideoCapture(videoPath);

        if (!cap.isOpened()) {
            System.out.println("Error: Cannot open video file");
            return;
        }

        Mat frame = new Mat();
        double fps = cap.get(Videoio.CAP_PROP_FPS);

        while (cap.read(frame)) {
            detectAndEstimateSpeed(frame, net, classes, fps);
            if (Imgcodecs.imwrite("output_frame.jpg", frame)) {
                System.out.println("Frame saved successfully.");
            }
        }

        cap.release();
    }

    // Load the class names from coco.names
    public static List<String> loadClasses(String namesFile) {
        List<String> classNames = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(namesFile));
            String line;
            while ((line = br.readLine()) != null) {
                classNames.add(line);
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classNames;
    }

    // Detect vehicles and estimate their speed
    public static void detectAndEstimateSpeed(Mat frame, Net net, List<String> classes, double fps) {
        Mat blob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        List<Mat> result = new ArrayList<>();
        List<String> layerNames = getOutputLayerNames(net);
        net.forward(result, layerNames);

        List<Rect> boxes = new ArrayList<>();
        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();

        for (Mat detection : result) {
            for (int i = 0; i < detection.rows(); i++) {
                Mat row = detection.row(i);
                Mat scores = row.colRange(5, row.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;

                if (confidence > 0.5) {
                    int classId = (int) mm.maxLoc.x;
                    if (classes.get(classId).equals("car")) {
                        int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                        int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                        int width = (int) (row.get(0, 2)[0] * frame.cols());
                        int height = (int) (row.get(0, 3)[0] * frame.rows());

                        int x = centerX - width / 2;
                        int y = centerY - height / 2;
                        boxes.add(new Rect(x, y, width, height));
                        classIds.add(classId);
                        confidences.add(confidence);
                    }
                }
            }
        }

        // Perform Non-Maximum Suppression to remove overlapping boxes
        MatOfRect rects = new MatOfRect();
        MatOfFloat confidencesMat = new MatOfFloat();
        rects.fromList(boxes);
        confidencesMat.fromList(confidences);

        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(rects.toList(), confidencesMat.toList(), 0.5f, 0.4f, indices);

        // Draw bounding boxes and estimate speed
        for (int idx : indices.toArray()) {
            Rect box = boxes.get(idx);
            Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 255, 0), 2);

            // Speed estimation (simple approach based on bounding box displacement)
            String speedText = "Speed: 50 km/h"; // Placeholder (improve by tracking over frames)
            Imgproc.putText(frame, speedText, box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 2);
        }
    }

    // Get the names of the YOLO output layers
    public static List<String> getOutputLayerNames(Net net) {
        List<String> layerNames = net.getLayerNames();
        MatOfInt outLayers = net.getUnconnectedOutLayers();
        List<String> outputLayerNames = new ArrayList<>();
        for (int i = 0; i < outLayers.size(0); i++) {
            outputLayerNames.add(layerNames.get(outLayers.get(i, 0)[0] - 1));
        }
        return outputLayerNames;
    }
}
