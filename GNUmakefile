CMAKE_ALT1 := /usr/local/bin/cmake
CMAKE_ALT2 := /Applications/CMake.app/Contents/bin/cmake
CMAKE := $(shell \
	which cmake 2>/dev/null || \
	([ -e ${CMAKE_ALT1} ] && echo "${CMAKE_ALT1}") || \
	([ -e ${CMAKE_ALT2} ] && echo "${CMAKE_ALT2}") \
	)

all: Release


Debug: build
	(cd build && ${CMAKE} -DCMAKE_BUILD_TYPE=$@ .. && make)

MinSizeRel: build
	(cd build && ${CMAKE} -DCMAKE_BUILD_TYPE=$@ .. && make)

Release: build
	(cd build && ${CMAKE} -DCMAKE_BUILD_TYPE=$@ .. && make)

RelWithDebugInfo: build
	(cd build && ${CMAKE} -DCMAKE_BUILD_TYPE=$@ .. && make)


run:
	build/bin/cis565_path_tracer scenes/cornell.json

build:
	mkdir -p build

clean:
	((cd build && make clean) 2>&- || true)

SCENE ?= scenes/cornell.json
REPORT_DIR := build/reports

run:
	build/bin/cis565_path_tracer $(SCENE)

nsys_profile: Release $(REPORT_DIR)
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	REPORT_FILE=$(REPORT_DIR)/nsys_report_$$TIMESTAMP; \
	echo "Saving profile to $$REPORT_FILE.qdrep"; \
	nsys profile --output $$REPORT_FILE \
		--trace=cuda,nvtx,osrt \
		--sample=none \
		build/bin/cis565_path_tracer $(SCENE)

$(REPORT_DIR):
	mkdir -p $(REPORT_DIR)

.PHONY: all Debug MinSizeRel Release RelWithDebugInfo clean nsys_profile
