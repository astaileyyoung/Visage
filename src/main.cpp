#include <iostream>
#include <chrono>

#include <torch/torch.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <cuda_runtime_api.h>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <indicators/progress_bar.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h" // for console logging

#include <utils.hpp>
#include <pipeline.hpp>


spdlog::level::level_enum parse_log_level(const std::string& level) {
    if (level == "trace") return spdlog::level::trace;
    if (level == "debug") return spdlog::level::debug;
    if (level == "info")  return spdlog::level::info;
    if (level == "warn")  return spdlog::level::warn;
    if (level == "error") return spdlog::level::err;
    if (level == "critical") return spdlog::level::critical;
    if (level == "off")   return spdlog::level::off;
    return spdlog::level::info; // default
}


void setup_logging(spdlog::level::level_enum log_level) {
    // 'static' means this variable is created only ONCE and keeps its value
    // between function calls. This is our "gatekeeper".
    static bool logging_is_initialized = false;

    // If we've already run this function once, just exit immediately.
    if (logging_is_initialized) {
        return;
    }
    
    // Mark that we have run the setup.
    logging_is_initialized = true;

    // Now, create all your loggers here. This block of code will now
    // only ever execute ONCE in the entire life of your program.
    auto main_logger = spdlog::stdout_color_mt("main");
    auto processor_logger = spdlog::stdout_color_mt("image_processor");

    spdlog::set_level(log_level);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
}


int main(int argc, char* argv[]) {
    int frameskip = 1;
    std::string log_level = "info";
    std::string src = argv[1];
    std::string dst = argv[2];
    bool show = false;
    if (argc > 3) {
        frameskip = std::stoi(argv[3]);
    }
    if (argc > 4) {
        log_level = argv[4];
    }
    if (argc > 5) {
        printf("\n\nshow: %s", argv[5]);
        if (std::string(argv[5]) == "-show") {
            show = true;
        }
    }

    setup_logging(parse_log_level(log_level));
    auto logger = spdlog::get("main");
    
    logger->debug("Num args: {}", argc);

    PreprocessParams detection_params;
    detection_params.mode = PreprocessingMode::LETTERBOX;
    detection_params.target_width = 640;
    detection_params.target_height = 640;
    detection_params.normalize = true;

    PreprocessParams recognition_params;
    recognition_params.mode = PreprocessingMode::DIRECT_RESIZE;
    recognition_params.target_width = 160;
    recognition_params.target_height = 160;
    recognition_params.normalize = true;

    cv::VideoCapture cap_info = cv::VideoCapture(src);
    if (!cap_info.isOpened()) {
        logger->error("Failed to open video at {} with cv::VideoCapture.", src);
        return -1;
    }
    int total_frames = cap_info.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = cap_info.get(cv::CAP_PROP_FPS);

    logger->debug("Total frames: {}", total_frames);
    logger->debug("FPS: {}", fps);

    cv::Ptr<cv::cudacodec::VideoReader> cap = cv::cudacodec::createVideoReader(src);
    if (!cap) {
        logger->error("Failed to open video at {} with cv::cudacodec::createVideoReader.", src);
    }
    logger->info("Loaded video from: {}", src);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string model_path = "../data/yolov11m-face-dynamic.trt";
    InferencePipeline detector(model_path, spdlog::level::debug);
    logger->debug("Instanciated detector from: {}", model_path);

    std::string embedding_model_path = "../data/facenet-dynamic.trt";
    RecognitionPipeline embedder(embedding_model_path);
    logger->debug("Instanciated embedder from: {}", embedding_model_path);

    int current_frame = 1;
    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PostfixText{"Processing video..."},
        indicators::option::ForegroundColor{indicators::Color::cyan},
        indicators::option::ShowPercentage{true},
        indicators::option::MaxProgress{total_frames}
    };

    auto start_time = std::chrono::steady_clock::now();
    logger->info("Starting detectionon {}", src);
    logger->debug("Frameskip: {}", frameskip);
    logger->debug("Show: {}", show);
    std::vector<Detection> all_detections;
    while (true) {
        std::vector<Detection> detections;

        cv::cuda::GpuMat gpuFrame;
        cap->nextFrame(gpuFrame);
        if (gpuFrame.empty()) break;

        if (current_frame % frameskip == 0) {
            detections = detector.run(gpuFrame, detection_params, current_frame, stream);
            if (detections.size() > 0) {
                std::vector<cv::Mat> embeddings = embedder.run(gpuFrame, detections, recognition_params, current_frame, stream);
                for (int i = 0; i < embeddings.size(); ++i) {
                    cv::Mat embedding = embeddings[i];
                    Detection det = detections[i];
                    det.embedding = embedding;
                    all_detections.push_back(det);
                }
            }
        }

        // Display image
        if (show){ 
            cv::Mat frame;
            gpuFrame.download(frame);
            if (frame.channels() == 4) {
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
            }
            if (detections.size() > 0) draw_detections(frame, detections, true);
            cv::imshow("frame", frame);
            if (cv::waitKey(fps) == 'q') {
                break;
            }
        }

        current_frame++;
        bar.set_progress(current_frame);
    }

    export_detections(all_detections, dst);

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    logger->info("Total runtime: {}", elapsed.count());

    return 0;
}