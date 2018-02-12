CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=OpenCRF_Main.cpp Config.cpp CRFModel.cpp Transmitter.cpp DataSet.cpp Util.cpp FactorGraph.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=OpenCRF_Main

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

