#include <iomanip>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaarithm.hpp>

#include <utils.hpp>
#include <ops.hpp>


cv::Mat resize_for_display(cv::Mat frame) {
    cv::resize(frame, frame, cv::Size(1920, 1080));
    return frame;
}


void print_blob(cv::Mat currentOutput) {
    std::cout << "---" << std::endl; // Separator for clarity
    std::cout << "  Dimensions: " << currentOutput.dims << std::endl;

    // Print the size of each dimension
    std::cout << "  Shape: (";
    for (int dim_idx = 0; dim_idx < currentOutput.dims; ++dim_idx) {
        std::cout << currentOutput.size[dim_idx] << (dim_idx == currentOutput.dims - 1 ? "" : "x");
    }
    std::cout << ")" << std::endl;
    std::cout << "  Type: " << currentOutput.type() << " (e.g., " << CV_32FC1 << " for float)" << std::endl;
    std::cout << "---" << std::endl; // End separator
}


std::string cleanRounding(float num, int decimals) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(decimals) << num;
    std::string result = stream.str();

    // Remove trailing zeros
    result.erase(result.find_last_not_of('0') + 1, std::string::npos);

    // If the last character is a '.', remove it too
    if (result.back() == '.') result.pop_back();

    return result;
}


void draw_detections(cv::Mat& frame, std::vector<Detection> detections, bool show_conf) {
    for (int i = 0; i < detections.size(); ++i) {
        Detection det = detections[i];
        int x1 = det.box[0];
        int y1 = det.box[1];
        int x2 = det.box[2];
        int y2 = det.box[3];
        float confidence = det.confidence;

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
        if (confidence != 0.0 && show_conf) {
            std::string conf_str = cleanRounding(confidence, 2);
            std::string label = "Conf: " + conf_str;
            cv::putText(frame, label, cv::Point(x2 + 5, y1 + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1); // Green text
        }
    }
}


cv::cuda::GpuMat extract_face(cv::cuda::GpuMat& img, Detection det) {
    int x1 = det.box[0];
    int y1 = det.box[1];
    int x2 = det.box[2];
    int y2 = det.box[3];

    int w = x2 - x1;
    int h = y2 - y1;
    cv::Rect face_rect(det.box[0], det.box[1], w, h);
    if (face_rect.x < 0 || face_rect.y < 0 || face_rect.x + face_rect.width > img.cols || face_rect.y + face_rect.height > img.rows) {
        std::cout << "Out of bounds." << std::endl;
        return cv::cuda::GpuMat(); // Return empty Mat if the rectangle is out of bounds
    }
    return img(face_rect);
}


void export_detections(const std::vector<Detection> detections, 
                       const std::string& filename,
                       const int rounding) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(9);
    file << "frame_num,face_num,x1,y1,x2,y2,confidence,embedding\n";
    for (int i = 0; i < detections.size(); ++i) {
        const Detection det = detections[i];
        file << det.frame_num << ',';
        file << det.face_num << ',';
        // Add coordinates
        for (int j = 0; j < 4; ++j) {
            file << det.box[j];
            if (j < 3) file << ",";
        }
        file << ",";
        file << cleanRounding(det.confidence, rounding);
        file <<",\"";

        torch::Tensor embedding_cpu = det.embedding.cpu().contiguous().view(-1);
        const float* embedding_ptr = embedding_cpu.data_ptr<float>();
        int embedding_size = embedding_cpu.numel();
        for (int j = 0; j < embedding_size; ++j) {
            file << embedding_ptr[j];
            if (j < embedding_size - 1) file << ",";
        }
        file << "\"\n";
    }
    file.close();
}