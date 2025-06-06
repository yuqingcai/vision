import QtQuick
import detectron2demo1

Window {
    id: mainWindow
    width: 640
    height: 480
    visible: true
    title: qsTr("detectron2 demo1")

    ImageDrawer {
        anchors.fill: parent
    }

}
