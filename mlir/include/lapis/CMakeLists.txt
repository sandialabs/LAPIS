add_subdirectory(Dialect)

# Copy LAPIS_config.h.in to LAPIS_config.h in the build's
# include directory, with #cmakedefine lines replaced with #define
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/LAPIS_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/LAPIS_config.h
  )

