# Locates the tensorFlow libraries and include directories.

include(FindPackageHandleStandardArgs)
unset(TFlite_FOUND)

find_library(TFlite_LIBRARY
    NAMES tensorflowlite
    HINTS ${AFFECTIVA_SDK_DIR}/lib/arm64-v8a)

find_library(TFlite_LIBRARY_DEBUG
    NAMES tensorflowlite
    HINTS ${AFFECTIVA_SDK_DIR}/lib/arm64-v8a)

# set TFlite_FOUND
find_package_handle_standard_args(TFlite DEFAULT_MSG )

# set external variables for usage in CMakeLists.txt
if(TFlite_FOUND)
    set(TFlite_LIBRARIES ${TFlite_LIBRARY})
endif()
