INFO = -Minfo=all
LIBS = -I/usr/local/cuda/12.3/targets/x86_64-linux/include/ -lboost_program_options
GPU = -acc=gpu
CXX = pgc++

all: main
  
main:
	$(CXX) -g $(GPU) $(INFO) $(LIBS) -o $@ main.cpp -L/usr/local/cuda/12.3/lib64 -lnvToolsExt -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib -lcublas -lcudart

clean:
	rm -f main
