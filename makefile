# Compilador e flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iutils -Isrc

# Diretórios
BINDIR = bin
BUILDDIR = build

# Arquivos
SRCS = $(wildcard src/*.cpp)
OBJS = $(patsubst src/%.cpp,$(BINDIR)/%.o,$(SRCS))
TARGET = $(BUILDDIR)/main

# Regra padrão
all: $(TARGET)

# Linkagem final (gera executável em build/)
$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilar .cpp em .o dentro de bin/
$(BINDIR)/%.o: src/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Garante que pastas existam
$(BINDIR):
	mkdir -p $(BINDIR)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Limpeza
clean:
	rm -f $(BINDIR)/*.o $(TARGET)

.PHONY: all clean
