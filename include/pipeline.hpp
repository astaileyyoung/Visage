#pragma once

#include <trt_infer.hpp>
#include <image_processor.hpp>


class InferencePipeline {
    public:
        InferencePipeline(const std::string& model_path, 
                 spdlog::level::level_enum log_level=spdlog::level::err);

        std::vector<Detection> run(const cv::cuda::GpuMat& img, 
                                   const PreprocessParams& params,
                                   int frame_num = 0,
                                   cudaStream_t stream = 0);

    private:
        TRTInfer trt;
        ImageProcessor proc;
};


class RecognitionPipeline {
    public:
        RecognitionPipeline(const std::string& model_path, 
                 spdlog::level::level_enum log_level=spdlog::level::err);

        std::vector<torch::Tensor> run(cv::cuda::GpuMat& img, 
                    std::vector<Detection> detections,
                    const PreprocessParams& params,
                    int frame_num = 0,
                    cudaStream_t stream = 0);

    private:
        TRTInfer trt;
        ImageProcessor proc;
};