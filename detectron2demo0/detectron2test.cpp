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

int main(int argc, char *argv[])
{
    const char* filePath = "../mask_rcnn_R_50_FPN_3x.pt";
    if (!std::filesystem::exists(filePath)) {
        printf("load model error: %s\n", filePath);
        return -1;
    }

    vision::detail::_register_ops();
    torch::jit::script::Module module = torch::jit::load(filePath);

    cv::Mat img = cv::imread("../../dataset/coco/train2017/000000000049.jpg");
    if (img.empty()) {
        printf("cannot open image file\n");
        return -1;
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    torch::Tensor img_tensor = torch::from_blob(
        img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();
    img_tensor = img_tensor.permute({2, 0, 1}).unsqueeze(0);

    c10::Dict<std::string, torch::Tensor> image_dict;
    image_dict.insert("image", img_tensor[0]);
    auto inputs = std::make_tuple(image_dict);
    auto output = module.forward({inputs});

    if (output.isList()) {
        auto out_list = output.toList();
        for (size_t i = 0; i < out_list.size(); i ++) {
            auto elem = out_list.get(i);
            std::cout << elem << std::endl;

            if (elem.isGenericDict()) {
                auto dict = elem.toGenericDict();
                if (dict.contains("pred_boxes")) {
                }

                if (dict.contains("scores")) {
                }

                if (dict.contains("pred_classes")) {
                }

                if (dict.contains("pred_masks")) {
                }
            }
        }
    }
    
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imshow("result", img);
    cv::waitKey(0);
    return 0;
}
