# Define variables for easier modification
CC = gcc
CXX = g++
# -w suppresses warnings - There are many warnings in sgb codes
CFLAGS = -w -O2
CXXFLAGS = -O2
LDFLAGS =
CTANGLE = ctangle # Command for ctangle

BIN_DIR = bin
SGB_DIR = sgb
TARGET = rudy

# --- Control Suffix Rules ---
# Clear all built-in suffix rules
.SUFFIXES:
# Define only the suffixes we care about for rule chaining
.SUFFIXES: .c .cpp .o .w .h

# Define VPATH: Where to look for source files (sgb directory and current directory)
# VPATH helps find gauss.c, rudy.cpp, and the generated .c/.h files in sgb/
VPATH = $(SGB_DIR):.

# --- SGB File Identification ---
# Find all relevant .w files in the sgb directory
_SGB_W_FILES_ALL = $(wildcard $(SGB_DIR)/gb_*.w)
# Define the specific .w files to exclude (These *must* still exist in sgb/ for includes)
_SGB_W_EXCLUDE = $(SGB_DIR)/boilerplate.w $(SGB_DIR)/gb_types.w
# Filter out the excluded files FROM BEING PROCESSED DIRECTLY
SGB_W_FILES = $(filter-out $(_SGB_W_EXCLUDE),$(_SGB_W_FILES_ALL))

# Define the corresponding .c and .h files that *will* be generated
SGB_GENERATED_C = $(patsubst $(SGB_DIR)/%.w,$(SGB_DIR)/%.c,$(SGB_W_FILES))
SGB_GENERATED_H = $(patsubst $(SGB_DIR)/%.w,$(SGB_DIR)/%.h,$(SGB_W_FILES))
SGB_GENERATED_FILES = $(wildcard $(SGB_DIR)/*.c) $(SGB_GENERATED_H)

# --- Source File Definitions ---
# Define C sources based on expected generated files + root files
C_SOURCES_SGB = $(SGB_GENERATED_C) # Full paths like sgb/gb_basic.c
C_SOURCES_ROOT = $(wildcard gauss.c) # Full path like gauss.c
# Get base names for object file calculation
C_SOURCES_ALL_BASES = $(notdir $(C_SOURCES_SGB)) $(notdir $(C_SOURCES_ROOT))

CPP_SOURCES = $(wildcard rudy.cpp)
CPP_SOURCES_BASE = $(notdir $(CPP_SOURCES))

# --- Object File Definitions ---
# Generate object file names based on the expected C and CPP source base names
C_OBJECTS = $(patsubst %.c,$(BIN_DIR)/%.o,$(C_SOURCES_ALL_BASES))
CPP_OBJECTS = $(patsubst %.cpp,$(BIN_DIR)/%.o,$(CPP_SOURCES_BASE))

# Combine all object files
OBJECTS = $(C_OBJECTS) $(CPP_OBJECTS)

# We want to keep these files to allow make run also without ctangle
.PRECIOUS: $(SGB_GENERATED_C) $(SGB_GENERATED_H)

# --- Build Rules ---

# Default target: build the executable
all: $(BIN_DIR)/$(TARGET)

# Rule to create the binary directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule 1: Generate SGB .c and .h files from .w files
# Target: sgb/gb_basic.c, sgb/gb_basic.h
# Prereq: sgb/gb_basic.w
$(SGB_DIR)/gb_%.c $(SGB_DIR)/gb_%.h: $(SGB_DIR)/gb_%.w
	@echo "Generating C/H for $(notdir $<) in $(SGB_DIR)..."
	cd $(SGB_DIR) && $(CTANGLE) $(notdir $<)

# Rule 2: Compile SGB .c files into .o files
# Target: bin/gb_basic.o
# Prereq: sgb/gb_basic.c (Explicit path)
# This rule is specific to SGB object files.
$(BIN_DIR)/gb_%.o: $(SGB_DIR)/gb_%.c $(BIN_DIR) $(SGB_GENERATED_H)
	@echo "Compiling SGB source $< -> $@"
	$(CC) $(CFLAGS) -I$(SGB_DIR) -c $< -o $@

# Rule 3: Compile non-SGB C files (gauss.c)
# Target: bin/gauss.o
# Prereq: gauss.c (Found via VPATH)
# This rule is generic but won't match gb_%.o because Rule 2 is more specific.
$(BIN_DIR)/%.o: %.c $(BIN_DIR)
	@echo "Compiling non-SGB C source $< -> $@"
	$(CC) $(CFLAGS) -I$(SGB_DIR) -c $< -o $@

# Rule 4: Compile C++ files
# Target: bin/rudy.o
# Prereq: rudy.cpp (Found via VPATH)
$(BIN_DIR)/%.o: %.cpp $(BIN_DIR)
	@echo "Compiling C++ source $< -> $@"
	$(CXX) $(CXXFLAGS) -I$(SGB_DIR) -c $< -o $@

# Rule 5: Link the executable and remove intermediate files
# Depends on all object files
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "Linking $@"
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@
	rm -f $(BIN_DIR)/*.o
	rm -f $(SGB_DIR)/test*.c
	@echo "Make complete."

# --- Cleanup Rule ---
clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)
	rm -f *.o # Remove any stray .o files in root, just in case
	# Also remove generated files from sgb directory
	@echo "Removing generated test SGB files..."
	rm -f $(SGB_DIR)/test*.c
	@echo "Cleanup complete."

# Phony targets are not actual files
.PHONY: all clean
