CC	 = mpicc
CXX	 = mpiCC
LD	 = mpiCC -hybrid
CHFLAGS	 = -host -msimd -O3
OBJDIR   = ./build
OBJECTS  = $(OBJDIR)/main.o $(OBJDIR)/Layer.o $(OBJDIR)/myMath.o
INCLUDES = -I./inc
LIBS	 =
VPATH    = ./src

all: main.o Layer.o myMath.o test_dnn

test_dnn: $(OBJECTS) | dir
	$(LD) -o $(OBJDIR)/test_dnn $^ $(LIBS)
main.o: main.cpp | dir
	$(CXX) $(CHFLAGS) $(INCLUDES) -c $< -o $(OBJDIR)/main.o
Layer.o: Layer.cpp | dir
	$(CXX) $(CHFLAGS) $(INCLUDES) -c $< -o $(OBJDIR)/Layer.o
myMath.o: myMath.c | dir
	$(CC) $(CHFLAGS) $(INCLUDES) -c $< -o $(OBJDIR)/myMath.o

dir:
	@mkdir -p $(OBJDIR)

.PHONY: clean
clean:
	rm -rf ./build
