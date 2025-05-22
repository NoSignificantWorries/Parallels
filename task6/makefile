INFO = -Minfo=all
LIBS = -I/usr/local/cuda/12.3/targets/x86_64-linux/include/ -lboost_program_options
GPU = -acc=gpu
CXX = pgc++

all: main
  
main:
	$(CXX) -g $(GPU) $(INFO) $(LIBS) -o $@ main.cpp -L/usr/local/cuda/12.3/lib64 -lnvToolsExt

clean:
	rm -f main
