CC      = g++
CFLAGS  = -g
LDFLAGS	= -lm
SRCS	= src\utils\c++\estimator.cpp src\utils\c++\nodes.cpp src\utils\c++\utility.cpp
OBJS	= src\utils\execution\estimator.o src\utils\execution\nodes.o src\utils\execution\utility.o

default: train

train: src\utils\c++\train.cpp $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o src\train.exe $< $(OBJS)

src\utils\execution\estimator.o: src\utils\c++\estimator.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

src\utils\execution\nodes.o: src\utils\c++\nodes.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

src\utils\execution\utility.o: src\utils\c++\utility.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	del src\utils\execution\*.o src\train.exe