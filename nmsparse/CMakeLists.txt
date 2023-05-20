file(GLOB TMP *.cu)
list(APPEND DIR_SRCS ${TMP})

add_library(nmsparse SHARED ${DIR_SRCS})
target_link_libraries(nmsparse ${NMSPARSE_LIBS})
