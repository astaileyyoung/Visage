#include <ops.hpp>
#include <utils.hpp>
#include <pipeline.hpp>
#include <trt_infer.hpp>


InferencePipeline::InferencePipeline(const std::string& model_path,
                   spdlog::level::level_enum log_level)
                   : trt(model_path), proc(log_level) {}

std::vector<Detection> InferencePipeline::run(const cv::cuda::GpuMat& img,
                                     const PreprocessParams& params,
                                     int frame_num,
                                     cudaStream_t stream) {
    float* buffer = static_cast<float*>(trt.getDeviceBuffer());
    proc.preprocess(img, params, buffer, stream);

    int batch = 1;
    int c = 3;
    int w = params.target_width;
    int h = params.target_height;

    std::vector<cv::Mat> raw_output = trt.infer(batch, c, h, w, stream);
    Predictions pred = {raw_output[0], frame_num};

    return proc.postprocess(pred);
}


RecognitionPipeline::RecognitionPipeline(const std::string& model_path,
                                         spdlog::level::level_enum log_level)
                                         : trt(model_path), proc(log_level) {}

std::vector<cv::Mat> RecognitionPipeline::run(cv::cuda::GpuMat& img,
                                 std::vector<Detection> detections,
                                 const PreprocessParams& params,
                                 int frame_num,
                                 cudaStream_t stream) {
    float* buffer = static_cast<float*>(trt.getDeviceBuffer());

    std::vector<cv::Mat> raw_output;

    int batch = detections.size();
    if (batch > 0) {
        int c = 3;
        int w = params.target_width;
        int h = params.target_height;

        int inputNumel = trt.getInputNumel();
        int64_t single_input_numel = inputNumel / batch;

        for (int i = 0; i < batch; ++i) {
            Detection det = detections[i];
            cv::cuda::GpuMat face = extract_face(img, det);
            proc.preprocess(img, params, buffer + i * single_input_numel, stream);
        }
        raw_output = trt.infer(batch, h, w, c, stream);
    }
    return raw_output;
}