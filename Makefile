GPU=1
OPENMP=1
DEBUG=1

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

SLIB=libjavernn.so
ALIB=libjavernn.a
EXEC=javernn
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


ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif


#srcs
LIB_CPP += $(wildcard ./src/javernn/*.cpp)
LIB_CPP += $(wildcard ./src/javernn/util/*.cpp)
OBJ_CPP += $(wildcard ./src/tools/*.cpp)
ifeq ($(GPU), 1) 
LDFLAGS += -lstdc++ 
OBJ_CU  += $(wildcard ./src/*.cu)
endif

#objs
OBJ_LIB := $(patsubst %.cpp, %.o, $(LIB_CPP))
OBJ_LIB += $(patsubst %.cu, %.o, $(OBJ_CU))
OBJ_BIN += $(patsubst %.cpp, %.o, $(OBJ_CPP))
EXECOBJA=javernn.o
$(info "test......")
$(info $(OBJ_LIB))
$(info $(OBJ_BIN))

OBJ_LIBS = $(addprefix $(OBJDIR), $(OBJ_LIB))
OBJ_BINS = $(addprefix $(OBJDIR), $(OBJ_BIN))
$(info $(OBJ_LIBS))
$(info $(OBJ_BINS))

DIRS := obj obj/src obj/src/tools obj/src/javernn  obj/src/javernn/util
DIRS += backup results 
all: $(DIRS) $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(OBJ_BINS) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJ_LIBS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJ_LIBS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu 
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(DIRS): 
	mkdir -p $@
obj_src:
	mkdir -p obj/src
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

