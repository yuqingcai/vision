cmake_minimum_required(VERSION 3.16)

project(detectron2demo1 VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 取消Qt的关键字宏定义，比如signals，slots等，这些宏定义会和第三方库的函数/变量名冲突。
# 取消以后 signals 变成 Q_SIGNALS, slots 变成 Q_SLOTS。
add_compile_definitions(QT_NO_KEYWORDS)

find_package(Qt6 6.8 REQUIRED COMPONENTS Core Gui Qml Quick)

qt_standard_project_setup(REQUIRES 6.8)

qt_add_executable(appdetectron2demo1
    main.cpp
)

qt_add_qml_module(appdetectron2demo1
    URI detectron2demo1
    VERSION 1.0
    QML_FILES
        Main.qml
        SOURCES imagedrawer.cpp imagedrawer.h
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/qml/detectron2demo1
)

target_include_directories(appdetectron2demo1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorch/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorchvision/include
)

add_library(torch SHARED IMPORTED GLOBAL)
set_target_properties(torch PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorch/lib/libtorch.dylib"
)

add_library(torch_cpu SHARED IMPORTED GLOBAL)
set_target_properties(torch_cpu PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorch/lib/libtorch_cpu.dylib"
)

add_library(c10 SHARED IMPORTED GLOBAL)
set_target_properties(c10 PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorch/lib/libc10.dylib"
)

add_library(torchvision SHARED IMPORTED GLOBAL)
set_target_properties(torchvision PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorchvision/lib/libtorchvision.dylib"
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appdetectron2demo1 PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appdetectron2demo1
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_link_libraries(appdetectron2demo1
    PRIVATE Qt6::Quick
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
    torch
    torch_cpu
    c10
    torchvision
)

set_target_properties(appdetectron2demo1 PROPERTIES
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorch/lib;${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorchvision/lib"
)

include(GNUInstallDirs)
install(TARGETS appdetectron2demo1
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 吧 model 目录打包到 package 里
add_custom_command(TARGET appdetectron2demo1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/model"
    "$<TARGET_FILE_DIR:appdetectron2demo1>/model"
)
