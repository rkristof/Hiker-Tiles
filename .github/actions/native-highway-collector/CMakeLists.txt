cmake_minimum_required(VERSION 3.16)
project(native_highway_collector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(OSMIUM_INCLUDE_DIR osmium/io/reader.hpp REQUIRED)
find_path(PROTOZERO_INCLUDE_DIR protozero/exception.hpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(EXPAT REQUIRED)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

add_executable(native-highway-collector main.cpp)
target_include_directories(native-highway-collector PRIVATE ${OSMIUM_INCLUDE_DIR} ${PROTOZERO_INCLUDE_DIR})
target_link_libraries(
	native-highway-collector
	PRIVATE
		ZLIB::ZLIB
		BZip2::BZip2
		EXPAT::EXPAT
		${LZ4_LIBRARY}
)
target_compile_options(native-highway-collector PRIVATE -O3 -Wall -Wextra -Wpedantic)
