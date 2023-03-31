SHOW =

MIRACL_DIR = MIRACL-CUDA/src
SM9_DIR = sm9
CUDA_DIR = cuda

MIRACL_CU = $(wildcard $(MIRACL_DIR)/*.cu)
SM9_CU = $(wildcard $(SM9_DIR)/*.cu)
CUDA_CU = $(wildcard $(CUDA_DIR)/*.cu)
SM9TEST_CU = $(wildcard SM9Test.cu)

SRC_CU = $(MIRACL_CU) $(SM9_CU) $(CUDA_CU) $(SM9TEST_CU)
OBJ_CU = $(patsubst %.cu, %.o, $(SRC_CU))

BUILD_DIR = ./build

#OPTION = -w -O3 -Xptxas -O3 -rdc=true# https://www.coder.work/article/6603658
OPTION = -w -rdc=true -O3 -Xptxas -dlcm=cg -O3

CC = nvcc

all:HUST-SM9

HUST-SM9:$(BUILD_DIR) $(OBJ_CU)
	$(SHOW)$(CC) $(OPTION) -o HUST-SM9 $(addprefix $(BUILD_DIR)/, $(OBJ_CU))

$(BUILD_DIR):
	$(SHOW)mkdir -p $(BUILD_DIR)
	$(SHOW)mkdir -p $(BUILD_DIR)/sm9
	$(SHOW)mkdir -p $(BUILD_DIR)/miracl
	$(SHOW)mkdir -p $(BUILD_DIR)/cuda

$(OBJ_CU): %.o: %.cu
	$(SHOW)$(CC) $(OPTION) -c $< -o $(BUILD_DIR)/$@

.PHONY:
	clean
clean:
	$(SHOW)rm -rd $(BUILD_DIR)
	$(SHOW)rm -f HUST-SM9