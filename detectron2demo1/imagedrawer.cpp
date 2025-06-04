#include "imagedrawer.h"
#include <iostream>

ImageDrawer::ImageDrawer()
{
    setFlag(QQuickItem::ItemHasContents, true);

    if (!std::filesystem::exists(m_modelPath)) {
        printf("load model error: %s\n", m_modelPath);
        return;
    }

    vision::detail::_register_ops();
    m_module = torch::jit::load(m_modelPath);

    m_image = QImage(m_imagePath);

    torch::Tensor imageT = imageToTensor();
    std::cout << "Tensor shape: " << imageT.sizes() << std::endl;

    c10::Dict<std::string, torch::Tensor> dict;
    dict.insert("image", imageT.squeeze());
    auto input = std::make_tuple(dict);
    auto output = m_module.forward({input});
    if (output.isList()) {
        auto out_list = output.toList();
        for (size_t i = 0; i < out_list.size(); i ++) {
            auto elem = out_list.get(i);
            std::cout << elem << std::endl;

            if (elem.isGenericDict()) {
                auto dict = elem.toGenericDict();
                if (dict.contains("pred_boxes")) {
                    std::cout << "pred_boxes:" << std::endl;
                }

                if (dict.contains("scores")) {
                    std::cout << "scores:" << std::endl;
                }

                if (dict.contains("pred_classes")) {
                    std::cout << "pred_classes:" << std::endl;
                }

                if (dict.contains("pred_masks")) {
                    std::cout << "pred_masks:" << std::endl;
                }
            }
        }
    }
}


torch::Tensor ImageDrawer::imageToTensor()
{
    QImage rgb = m_image.convertToFormat(QImage::Format_RGB888);
    int w = rgb.width();
    int h = rgb.height();

    torch::Tensor tensor = torch::from_blob(
        rgb.bits(),
        { h, w, 3 },
        torch::kByte
        );

    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
    // 维度变成[B, C, H, W]([1, 3, H, W])
    tensor = tensor.unsqueeze(0);

    return tensor;
}


void ImageDrawer::geometryChange(const QRectF &newGeometry,
                                 const QRectF &oldGeometry)
{
    QQuickItem::geometryChange(newGeometry, oldGeometry);
    update();
}


QSGNode* ImageDrawer::updatePaintNode(QSGNode *oldNode,
                                      UpdatePaintNodeData *)
{
    QSGSimpleTextureNode *node = static_cast<QSGSimpleTextureNode *>(oldNode);
    if (!node) {
        node = new QSGSimpleTextureNode();
    }

    if (m_image.isNull()) {
        return node;
    }

    QSGTexture *texture = window()->createTextureFromImage(m_image);
    node->setTexture(texture);

    QSize imageSize = m_image.size();
    QRectF targetRect(QPointF(0, 0), QSizeF(imageSize));
    node->setRect(targetRect);

    node->markDirty(QSGNode::DirtyMaterial);

    return node;
}
