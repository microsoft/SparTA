file(GLOB TMP_CUDA *.cu)
file(GLOB TMP_CUDA_HEADER *.cuh)
file(GLOB TMP_CPP *.cpp)
list(APPEND DIR_SRCS ${TMP_CUDA} ${TMP_CUDA_HEADER} ${TMP_CPP})

add_library(nmsparse SHARED ${DIR_SRCS})
target_link_libraries(nmsparse ${NMSPARSE_LIBS})
set_target_properties(nmsparse PROPERTIES LINKER_LANGUAGE CUDA)
