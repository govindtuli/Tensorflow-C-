
#include <iostream>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

int main() {
    // Load the model
    const std::string export_dir = "../model";
    tensorflow::SavedModelBundle bundle;

    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    std::unordered_set<std::string> tags = {"serve"};

    tensorflow::Status status = tensorflow::LoadSavedModel(session_options, run_options, export_dir, tags, &bundle);

    // Check if the model was loaded successfully
    if (!status.ok()) {
        std::cerr << "Error loading the model: " << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Model loaded successfully!" << std::endl;
    }

    return 0;
}


// Resizing

// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/public/session.h"
// #include "opencv2/opencv.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

// using namespace tensorflow;
// using namespace cv;

// Tensor ConvertImageToTensor(Mat img) {
//     // Convert OpenCV mat to TensorFlow Tensor
//     Tensor image_tensor(DT_FLOAT, TensorShape({img.rows, img.cols, img.channels()}));
//     auto image_tensor_mapped = image_tensor.tensor<float, 3>();

//     for (int y = 0; y < img.rows; ++y) {
//         for (int x = 0; x < img.cols; ++x) {
//             for (int c = 0; c < img.channels(); ++c) {
//                 image_tensor_mapped(y, x, c) = img.at<Vec3b>(y, x)[c] / 255.0;
//             }
//         }
//     }

//     return image_tensor;
// }

// Mat ResizeAndPad(Mat img, int target_height, int target_width) {
//     // Resize the image maintaining the aspect ratio
//     double aspect_ratio = static_cast<double>(img.cols) / img.rows;
//     int new_width = target_width;
//     int new_height = static_cast<int>(new_width / aspect_ratio);
//     if(new_height > target_height) {
//         new_height = target_height;
//         new_width = static_cast<int>(new_height * aspect_ratio);
//     }
    
//     resize(img, img, Size(new_width, new_height));
    
//     // Pad the image to make it target_width x target_height
//     int top = (target_height - new_height) / 2;
//     int bottom = target_height - top - new_height;
//     int left = (target_width - new_width) / 2;
//     int right = target_width - left - new_width;
    
//     copyMakeBorder(img, img, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
    
//     return img;
// }

// Mat GenerateRandomImage(int height, int width) {
//     // Create a random image
//     Mat img(height, width, CV_8UC3);
//     randu(img, Scalar::all(0), Scalar::all(255));

//     // Save the image
//     imwrite("image.jpg", img);

//     return img;
// }

// int main() {
//     // Generate a random image and save it as "image.jpg"
//     Mat img = GenerateRandomImage(256, 256);
//     std::cout << "Image generated and saved as \"image.jpg\"." << std::endl;

//     // Resize and pad the image
//     img = ResizeAndPad(img, 128, 128);
//     std::cout << "Image resized and padded to 128x128." << std::endl;

//     // Convert to TensorFlow Tensor
//     Tensor image_tensor = ConvertImageToTensor(img);
//     std::cout << "Image converted to TensorFlow Tensor." << std::endl;

//     // TODO: Run the Tensor through your model

//     return 0;
// }

// #include <iostream>
// #include <random>
// #include "opencv2/opencv.hpp"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/core/public/session.h"
// #include "tensorflow/core/framework/tensor.h"

// Convert cv::Mat to TensorFlow::Tensor


// #include <tensorflow/core/public/session.h>
// #include <tensorflow/core/platform/env.h>
// #include <tensorflow/cc/ops/standard_ops.h>
// #include <tensorflow/core/framework/tensor.h>
// #include <tensorflow/cc/framework/gradients.h>
// #include <tensorflow/cc/framework/ops.h>
// #include <tensorflow/cc/ops/standard_ops.h>
// #include <tensorflow/core/framework/types.h>
// #include <tensorflow/cc/client/client_session.h>
// #include <tensorflow/cc/ops/const_op.h>
// #include <tensorflow/cc/ops/array_ops.h>
// #include <opencv2/opencv.hpp>

// using namespace tensorflow;
// using namespace tensorflow::ops;
// using namespace std;

// tensorflow::Tensor MatToTensor(cv::Mat img) {
//     cv::Mat img_float;
//     img.convertTo(img_float, CV_32FC3);
//     img_float = img_float / 255.0;
//     auto img_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({img.rows, img.cols, img.channels()}));
//     auto tensor_mapped = img_tensor.tensor<float, 3>();

//     const float * source_ptr = (float*)img_float.data;
//     // copying the data into the corresponding tensor
//     for (int y = 0; y < img_float.rows; ++y) {
//         const float* source_row_ptr = source_ptr + (y * img_float.cols * img_float.channels());
//         for (int x = 0; x < img_float.cols; ++x) {
//             const float* source_pixel_ptr = source_row_ptr + (x * img_float.channels());
//             for (int c = 0; c < img_float.channels(); ++c) {
//                 const float* source_value_ptr = source_pixel_ptr + c;
//                 tensor_mapped(y, x, c) = *source_value_ptr;
//             }
//         }
//     }

//     return img_tensor;
// }


// cv::Mat ResizeAndPad(cv::Mat img, int new_height, int new_width, bool pad = false, int interpolation = cv::INTER_LINEAR) {
//     cv::Size old_size = img.size();

//     float ratio = std::min(new_height * 1.0 / old_size.height, new_width * 1.0 / old_size.width);
//     cv::Size new_size((int)(old_size.width * ratio), (int)(old_size.height * ratio));

//     cv::Mat img_new;
//     cv::resize(img, img_new, new_size, 0, 0, interpolation);

//     if (pad) {
//         int delta_w = new_width - new_size.width;
//         int delta_h = new_height - new_size.height;
//         int top = delta_h / 2;
//         int bottom = delta_h - top;
//         int left = delta_w / 2;
//         int right = delta_w - left;

//         cv::copyMakeBorder(img_new, img_new, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
//     }

//     return img_new;
// }

// cv::Mat PaddingOnly(cv::Mat o_img, int pixels) {
//     cv::Mat img = cv::Mat::zeros(cv::Size(pixels, pixels), CV_8UC3);

//     try {
//         int x_offset = (pixels - o_img.cols) / 2;
//         int y_offset = (pixels - o_img.rows) / 2;

//         cv::Mat roi = img(cv::Rect(x_offset, y_offset, o_img.cols, o_img.rows));
//         o_img.copyTo(roi);
//     }
//     catch (...) {
//         try {
//             cv::resize(o_img, o_img, cv::Size(pixels, o_img.rows * pixels / o_img.cols));
//             int x_offset = (pixels - o_img.cols) / 2;
//             int y_offset = (pixels - o_img.rows) / 2;

//             cv::Mat roi = img(cv::Rect(x_offset, y_offset, o_img.cols, o_img.rows));
//             o_img.copyTo(roi);
//         }
//         catch (...) {
//             cv::resize(o_img, o_img, cv::Size(o_img.cols * pixels / o_img.rows, pixels));
//             int x_offset = (pixels - o_img.cols) / 2;
//             int y_offset = (pixels - o_img.rows) / 2;

//             cv::Mat roi = img(cv::Rect(x_offset, y_offset, o_img.cols, o_img.rows));
//             o_img.copyTo(roi);
//         }
//     }

//     return img;
// }

// cv::Mat GenerateRandomImage(int height, int width) {
//     cv::Mat img(height, width, CV_8UC3);
//     cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
//     return img;
// }

// int main() {
//     cv::Mat img = cv::imread("/home/prodesk/tensorflow_cpp_test/build/new_img.jpg");
//     cv::imwrite("image.jpg", img);
//     img = ResizeAndPad(img, 128, 128, true);
//     tensorflow::Tensor img_tensor = MatToTensor(img);
//     std::cout << "Image tensor shape: " << img_tensor.shape().DebugString() << std::endl;
//     cv::imwrite("image.jpg", img);
//     return 0;
// }

// #include <opencv2/opencv.hpp>
// #include <opencv2/ximgproc.hpp>

// class StageOneSegmentation {
//     public:
//         cv::Ptr<cv::ximgproc::StructuredEdgeDetection> edge_detector;
//         cv::Ptr<cv::ximgproc::FastLineDetector> fld;

//         // constructor to load the edge detection model
//         StageOneSegmentation() {
//             edge_detector = cv::ximgproc::createStructuredEdgeDetection("../model.yml");
//             fld = cv::ximgproc::createFastLineDetector();
//         }

//         cv::Mat getSegmentMask(cv::Mat& image) {
//             cv::Mat edges;
//             cv::Mat edgeView, fldEdgeView;

//             // ensure the image is in the correct format
//             image.convertTo(image, CV_32FC3, 1.0 / 255.0);

//             // detect the edges
//             edge_detector->detectEdges(image, edges);

//             // convert the edges to grayscale
//             edges.convertTo(edges, CV_8UC1, 255.0);

//             // extract lines
//             std::vector<cv::Vec4f> lines;
//             fld->detect(edges, lines);

//             cv::cvtColor(edges, edgeView, cv::COLOR_GRAY2BGR);
//             fld->drawSegments(edgeView, lines, true);

//             // ensure the image is in the correct format for saving
//             edgeView.convertTo(fldEdgeView, CV_8UC3, 255.0);

//             return fldEdgeView;
//         }
// };

// int main() {
//     // create instance of the segmentation class
//     StageOneSegmentation stageOneSegmentation;

//     // load the image to segment
//     cv::Mat image = cv::imread("../scanned_image.png");

//     // get the segmentation mask
//     cv::Mat segmented = stageOneSegmentation.getSegmentMask(image);

//     // save the color segmentation mask to a file
//     cv::imwrite("segmented.jpg", segmented);

//     return 0;
// }


// #include <tensorflow/cc/ops/standard_ops.h>
// #include <tensorflow/cc/client/client_session.h>
// #include <tensorflow/core/framework/tensor.h>
// #include <tensorflow/core/lib/core/status.h>
// #include <tensorflow/cc/ops/const_op.h>
// #include <tensorflow/cc/ops/image_ops.h>
// #include <opencv2/opencv.hpp>
// #include <tensorflow/core/public/session.h>
// #include <tensorflow/cc/saved_model/loader.h>
// #include <tensorflow/cc/saved_model/tag_constants.h>

// class ImageSegmenter {
//     std::unique_ptr<tensorflow::Session> session;

// public:
//     ImageSegmenter(const std::string& model_file) {
//         tensorflow::SavedModelBundle bundle;
//         tensorflow::Status load_status = tensorflow::LoadSavedModel(tensorflow::SessionOptions(), 
//                                                                     tensorflow::RunOptions(), 
//                                                                     model_file, 
//                                                                     {tensorflow::kSavedModelTagServe}, 
//                                                                     &bundle);

//         if (!load_status.ok()) {
//             std::cerr << "Failed to load model: " << load_status;
//             return;
//         }

//         session = std::move(bundle.session);
//     }

//     tensorflow::Tensor cv_mat_to_tensor(cv::Mat img) {
//         tensorflow::Tensor img_tensor(
//             tensorflow::DT_FLOAT,
//             tensorflow::TensorShape({1, img.rows, img.cols, img.channels()}));

//         float* tensor_data_ptr = img_tensor.flat<float>().data();
//         cv::Mat img_float;
//         img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);
//         std::memcpy(tensor_data_ptr, img_float.data, img_float.total() * img_float.elemSize());

//         return img_tensor;
//     }

//     cv::Mat tensor_to_cv_mat(tensorflow::Tensor& tensor) {
//         auto shape = tensor.shape();
//         assert(shape.dims() == 4);
//         int height = shape.dim_size(1);
//         int width = shape.dim_size(2);

//         cv::Mat output_img = cv::Mat(height, width, CV_32FC1);
//         std::memcpy(output_img.data, tensor.flat<float>().data(), tensor.AllocatedBytes());

//         return output_img;
//     }

//     cv::Mat predict(const cv::Mat& img) {
//         tensorflow::Tensor img_tensor = cv_mat_to_tensor(img);
//         std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"dense_input", img_tensor}};

//         std::vector<tensorflow::Tensor> outputs;
//         tensorflow::Status run_status = session->Run(inputs, {"dense_1"}, {}, &outputs);

//         if (!run_status.ok()) {
//             std::cerr << "Failed to run model: " << run_status;
//             return cv::Mat();
//         }

//         cv::Mat output_img = tensor_to_cv_mat(outputs[0]);
//         return output_img;
//     }
// };

// int main() {
//     // Create the instance of the ImageSegmenter
//     ImageSegmenter segmenter("../saved_model");

//     // Load the image
//     cv::Mat img = cv::imread("../scanned_image.png");

//     // Make a prediction and get the result
//     cv::Mat result = segmenter.predict(img);

//     // Save the result
//     cv::imwrite("../seg_output.jpg", result * 255);

//     return 0;
// }



