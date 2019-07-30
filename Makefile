GPU=0
OPENMP=0
DEBUG=1
TEST=1

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
	  -gencode arch=compute_61,code=[sm_61,compute_61]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

SLIB=libGrape.so
ALIB=libGrape.a
EXEC=Grape
TEST_EXE=Test
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)
LDFLAGS += -lstdc++ 


ifeq ($(TEST), 1) 
COMMON+= -Isrc/test/include
LDFLAGS += -Lsrc/test/lib  -lgtest
TEST_CPP_SRCS := $(shell find ./src/test -name "*.cpp")
OBJ_TEST_SRCS := $(patsubst ./%.cpp, ./$(OBJDIR)%.o, $(TEST_CPP_SRCS))
TEST_H_SRCS := $(shell find ./src/test -maxdepth 0 -name "*.h" )
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
LIB_CU_SRCS := $(shell find ./src/grape -name "*.cu")
OBJ_LIB_CU  := $(patsubst ./%.cu, ./$(OBJDIR)%.o, $(LIB_CU_SRCS))
endif



#srcs
LIB_CPP_SRCS := $(shell find ./src/grape -name "*.cpp")
BIN_CPP_SRCS := $(shell find ./src/tools -name "*.cpp")

#objs
OBJ_LIB_CPP := $(patsubst ./%.cpp, ./$(OBJDIR)%.o, $(LIB_CPP_SRCS))
OBJ_BIN_CPP := $(patsubst ./%.cpp, ./$(OBJDIR)%.o, $(BIN_CPP_SRCS))


DIRS := obj obj/src obj/src/tools obj/src/grape   
DIRS += obj/src/grape/util obj/src/grape/op obj/src/grape/optimizer obj/src/grape/parse 
DIRS += backup results model 

ifeq ($(TEST), 1) 
	DIRS += obj/src/test
endif

all: $(DIRS) $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)

test: $(DIRS) $(TEST_EXE)

$(TEST_EXE):$(OBJ_TEST_SRCS) $(OBJ_LIB_CPP) $(TEST_H_SRCS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) 
	
$(EXEC): $(OBJ_BIN_CPP) $(ALIB) 
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJ_LIB_CPP)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJ_LIB_CPP)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu 
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(DIRS): 
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

