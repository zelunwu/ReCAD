add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE recad_cuda)

# Make the freshly built DLL available next to the test executable (Windows
# does not search the build tree for DLLs at runtime).
add_custom_command(TARGET test_kernels POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:recad_cuda> $<TARGET_FILE_DIR:test_kernels>)

add_test(NAME kernels COMMAND test_kernels)