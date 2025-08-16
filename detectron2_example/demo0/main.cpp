#include <cstdlib>
#include <cstdio>
#include <vector>
#include <filesystem>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;

cv::Mat eval_scripting_model(torch::jit::script::Module& module, 
    char* filePath, float scale)
{
    cv::Mat img = cv::imread(filePath);

    // cv::Mat img_small;
    // cv::resize(img, img_small, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    torch::Tensor imageT = torch::from_blob(
        img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();
    imageT = imageT.permute({2, 0, 1});
    c10::Dict<std::string, torch::Tensor> dict;
    dict.insert("image", imageT);
    auto output = module.forward({std::make_tuple(dict)});

    if (output.isList()) {
        auto out_list = output.toList();
        for (size_t i = 0; i < out_list.size(); i ++) {
            auto elem = out_list.get(i);

            if (elem.isGenericDict()) {
                auto dict = elem.toGenericDict();
                
                if (dict.contains("pred_boxes") && dict.contains("scores")) {
                    auto ivalPredBoxes = dict.at("pred_boxes");
                    auto ivalScores = dict.at("scores");

                    if (ivalPredBoxes.isTensor() && ivalScores.isTensor()) {
                        torch::Tensor boxesT = ivalPredBoxes.toTensor();
                        torch::Tensor scoresT = ivalScores.toTensor();
                        torch::Tensor keep = vision::ops::nms(boxesT, scoresT, 0.5);

                        for (int j = 0; j < keep.size(0); j ++) {
                            auto box = boxesT[j];
                            float x0 = box[0].item<float>();
                            float y0 = box[1].item<float>();
                            float x1 = box[2].item<float>();
                            float y1 = box[3].item<float>();
                            printf("box %d: [%.1f, %.1f, %.1f, %.1f]\n", j, x0, y0, x1, y1);
                            cv::rectangle(img, cv::Point(x0, y0), 
                                cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
                        }

                        if (dict.contains("pred_masks")) {
                            auto ivalPredMasks = dict.at("pred_masks");
                            if (ivalPredMasks.isTensor()) {
                                torch::Tensor masksT = ivalPredMasks.toTensor();
                                std::cout << masksT.sizes() << std::endl;

                                masksT = masksT.squeeze(1);
                                std::cout << masksT.sizes() << std::endl;
                                
                                for (int j = 0; j < keep.size(0); j ++) {
                                    int idx = keep[j].item<int>();
                                    torch::Tensor mask = masksT[idx];
                                    std::cout << mask.sizes() << std::endl;

                                    auto box = boxesT[j];
                                    int x0 = std::round(box[0].item<float>());
                                    int y0 = std::round(box[1].item<float>());
                                    int x1 = std::round(box[2].item<float>());
                                    int y1 = std::round(box[3].item<float>());
                                    int w = std::max(x1 - x0, 1);
                                    int h = std::max(y1 - y0, 1);
                                    
                                    // 二值化
                                    mask = (mask > 0.5).to(torch::kU8).cpu();
                                    cv::Mat mask_mat(mask.size(0), mask.size(1), 
                                        CV_8U, mask.data_ptr());

                                    // resize 到检测框大小
                                    cv::Mat mask_resized;
                                    cv::resize(mask_mat, mask_resized, cv::Size(w, h), 0, 0, 
                                        cv::INTER_LINEAR);
                                    cv::GaussianBlur(mask_resized, mask_resized, cv::Size(3, 3), 0);
                                    
                                    // 生成彩色掩码
                                    cv::Scalar color(128, 0, 0);
                                    for (int yy = 0; yy < h; ++yy) {
                                        for (int xx = 0; xx < w; ++xx) {
                                            if (mask_resized.at<uchar>(yy, xx)) {
                                                int yy_img = y0 + yy;
                                                int xx_img = x0 + xx;
                                                if (yy_img >= 0 && yy_img < img.rows &&
                                                    xx_img >= 0 && xx_img < img.cols) {
                                                    // 半透明叠加
                                                    for (int c = 0; c < 3; ++c) {
                                                        img.at<cv::Vec3b>(yy_img, xx_img)[c] =
                                                            uchar(0.8 * color[c] + 0.6 * img.at<cv::Vec3b>(yy_img, xx_img)[c]);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }
    }

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    return img;
}


cv::Mat eval_tracing_model(torch::jit::script::Module& module, char* filePath)
{
    cv::Mat img = cv::imread(filePath);

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    torch::Tensor imageT = torch::from_blob(
        img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();
    imageT = imageT.permute({2, 0, 1});
    auto output = module.forward({imageT});
    if (output.isTuple()) {
        auto out_tuple = output.toTuple();
        auto boxes_ivalue = out_tuple->elements()[0];

        torch::Tensor boxes = boxes_ivalue.toTensor();
        for (int j = 0; j < boxes.size(0); j ++) {
            auto box = boxes[j];
            float x0 = box[0].item<float>();
            float y0 = box[1].item<float>();
            float x1 = box[2].item<float>();
            float y1 = box[3].item<float>();
            // 你可以在这里画框或保存坐标
            printf("box %d: [%.1f, %.1f, %.1f, %.1f]\n", j, x0, y0, x1, y1);
            cv::rectangle(img, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
        }
    }
    
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    return img;
}


int main(int argc, char *argv[])
{
    char* filePath = argv[1];

    vision::detail::_register_ops();
    float scale = 1.0;

    // const char* modelTracingPath = "../mask_rcnn_R_50_FPN_3x_tracing.pt";
    // if (!std::filesystem::exists(modelTracingPath)) {
    //     printf("load model error: %s\n", modelTracingPath);
    //     return -1;
    // }
    // torch::jit::script::Module moduleTracing = torch::jit::load(modelTracingPath);
    // cv::Mat img0 = eval_tracing_model(moduleTracing, filePath);
    // cv::imshow("tracing module eval", img0);

    const char* modelScriptingPath = "../../model/mask_rcnn_R_50_FPN_3x_scripting.pt";
    if (!std::filesystem::exists(modelScriptingPath)) {
        printf("load model error: %s\n", modelScriptingPath);
        return -1;
    }
    torch::jit::script::Module moduleScripting = torch::jit::load(modelScriptingPath);
    cv::Mat img1 = eval_scripting_model(moduleScripting, filePath, scale);
    cv::imshow("scripting modeule eval", img1);

    cv::waitKey(0);

    return 0;
}
