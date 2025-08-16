#ifndef IMAGEDRAWER_H
#define IMAGEDRAWER_H


#include <torch/script.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

#include <QQuickItem>
#include <QImage>
#include <QSGSimpleTextureNode>
#include <QQuickWindow>

class ImageDrawer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(ImageDrawer)

public:
    ImageDrawer();

protected:
    QSGNode* updatePaintNode(
        QSGNode *oldNode,
        QQuickItem::UpdatePaintNodeData *updatePaintNodeData);


    void geometryChange(
        const QRectF &newGeometry,
        const QRectF &oldGeometry);

    torch::Tensor imageToTensor();

private:
    QImage m_image;
    const char* m_imagePath = "/Volumes/VolumeEXT/Project/NNDL/dataset/coco/test2017/000000000057.jpg";
    const char* m_modelPath = "./model/mask_rcnn_R_50_FPN_3x.pt";
    torch::jit::script::Module m_module;

Q_SIGNALS:
};

#endif // IMAGEDRAWER_H
