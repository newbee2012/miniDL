# Compiler settings
CXX = g++
CXXFLAGS = -Winline -Wfatal-errors -fexceptions -Iinclude -Iextern/include

# Project settings
SRCDIR = src
OBJDIR = obj/Debug
BINDIR = bin/Debug

ifeq ($(DEBUG), 1)
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O3
	OBJDIR = obj/Release
	BINDIR = bin/Release
endif

# Files and folders
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))
TARGET = $(BINDIR)/miniDL

# Libraries
LIBS = -Lextern/lib -lboost_system -lpthread -ljsoncpp
LDFLAGS = -O3 -static-libstdc++ -static -s $(LIBS)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(BINDIR)
	$(CXX) $^ $(LDFLAGS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)
