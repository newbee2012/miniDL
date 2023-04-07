# Compiler settings
CXX = g++
CXXFLAGS = -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include

# Project settings
SRCDIR = src
OBJDIR = obj/Release
BINDIR = bin/Release

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
