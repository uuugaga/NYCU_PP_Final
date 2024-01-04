CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -pg `pkg-config --cflags opencv4` -fopenmp
NVCCFLAGS = -Xptxas -v,--opt-level=3 -c
OPENCV := `pkg-config --libs opencv4`
CUDA_PATH := /usr/local/cuda
LDFLAGS = -lm -lpthread -L$(CUDA_PATH)/lib64 -lcudart $(OPENCV)

# Targets
TARGET_OMP = omp
TARGET_NO_OMP = sequential
TARGET_PTHREAD = pthread
TARGET_CUDA = cuda

# Source directories
SRC_DIR = ./src
OBJ_DIR = ./obj
CUDA_DIR = $(SRC_DIR)/cuda

# Source files
SRCS_OMP = $(SRC_DIR)/omp.cpp
SRCS_NO_OMP = $(SRC_DIR)/sequential.cpp
SRCS_PTHREAD = $(SRC_DIR)/pthread.cpp
SRCS_CUDA = $(CUDA_DIR)/cuda.cpp
CU_SRCS = $(CUDA_DIR)/kernel.cu

# Object files
OBJS_OMP = $(SRCS_OMP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJS_NO_OMP = $(SRCS_NO_OMP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJS_PTHREAD = $(SRCS_PTHREAD:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJS_CUDA = $(SRCS_CUDA:$(CUDA_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS = $(CU_SRCS:$(CUDA_DIR)/%.cu=$(OBJ_DIR)/%.o)

.PHONY: all run_omp run_pthread run_no_omp clean

all: $(TARGET_NO_OMP) $(TARGET_PTHREAD) $(TARGET_OMP) $(TARGET_CUDA)



cuda_test: $(TARGET_NO_OMP) $(TARGET_CUDA)
	bash ./script/cuda_test.sh

omp_test: $(TARGET_NO_OMP) $(TARGET_OMP)
	bash ./script/omp_test.sh

pthread_test: $(TARGET_NO_OMP) $(TARGET_PTHREAD)
	bash ./script/pthread_test.sh

$(TARGET_CUDA): $(OBJS_CUDA) $(CU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(CUDA_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV) -I$(CUDA_PATH)/include

$(OBJ_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -I$(CUDA_PATH)/include

$(TARGET_OMP): $(OBJS_OMP)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV) $(LDFLAGS)

$(TARGET_PTHREAD): $(OBJS_PTHREAD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV) $(LDFLAGS)

$(TARGET_NO_OMP): $(OBJS_NO_OMP)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV)

clean:
	rm -f $(TARGET_OMP) $(TARGET_NO_OMP) $(TARGET_PTHREAD) $(TARGET_CUDA) $(OBJS_OMP) $(OBJS_NO_OMP) $(OBJS_PTHREAD) $(OBJS_CUDA) $(CU_OBJS)
