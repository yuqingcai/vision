#------------------------------------------------------------------------------
#
#	LazyCat Design
#
#	Ethan Tsai
#	yuqingcai@gmail.com
#	2025-4-25
#
#------------------------------------------------------------------------------
CC = gcc
CXX = g++
AS = as
LD = ld
AR = ar
OBJCOPY	= objcopy
OBJDUMP	= objdump
INCLUDE_PATH = -I/usr/local/include -I./libtorch/include -I./libtorchvision/include -I/opt/homebrew/opt/opencv/include/opencv4
LIBRARY_PATH = -L/usr/local/lib/ -L./libtorch/lib -L./libtorchvision/lib -L/opt/homebrew/opt/opencv/lib
LIBS = -lz -ltorch -ltorch_cpu -ltorchvision -lc10 -lopencv_core \
		-lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

C_FLAGS = $(INCLUDE_PATH) -Os
CPP_FLAGS = $(INCLUDE_PATH) -Os
CXX_FLAGS = $(INCLUDE_PATH) -std=c++20 -Wall -Wextra -Os

%.o: %.c
	$(CC) $(C_FLAGS) -c -g $< -o $@
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c -g $< -o $@
%.o: %.cc
	$(CXX) $(CXX_FLAGS) -c -g $< -o $@
%.d: %.cpp
	$(CXX) $(CXX_FLAGS) -MM $< -MT $*.o > $@
%.d: %.cc
	$(CXX) $(CXX_FLAGS) -MM $< -MT $*.o > $@
%.d: %.c
	$(CC) $(C_FLAGS) -MM $< -MT $*.o > $@

targets: detectron2demo0

detectron2demo0: main.o
	$(CXX) $^ -o $@ $(LIBRARY_PATH) $(LIBS) \
	-Wl,-rpath,./libtorch/lib \
	-Wl,-rpath,./libtorchvision/lib \
	-Wl,-rpath,/opt/homebrew/opt/opencv/lib

clean:
	rm -f detectron2demo0
	rm -f *.o

