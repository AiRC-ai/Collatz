CMAKE ?= cmake
BUILD_DIR ?= build
CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic
LDFLAGS ?= -lm

.PHONY: all configure test clean legacy-c

all: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

test: all
	$(CMAKE) --build $(BUILD_DIR) --target test
	./$(BUILD_DIR)/collatz_validate_sources

legacy-c: collatz_plot.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
