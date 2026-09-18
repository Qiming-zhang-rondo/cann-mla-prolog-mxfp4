# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of this repository.

# The source checkout can omit Tensor API headers that are shipped with CANN.
# Both binary staging and package installation must use the same complete root.
function(resolve_tensor_api_headers)
  set(_tensor_api_candidates)
  if(TENSOR_API)
    list(APPEND _tensor_api_candidates "${TENSOR_API}")
  endif()
  if(OPTENSOR_SOURCE_PATH)
    list(APPEND _tensor_api_candidates "${OPTENSOR_SOURCE_PATH}/include/tensor_api")
  endif()
  foreach(_cann_root IN ITEMS "${ASCEND_DIR}" "${ASCEND_CANN_PACKAGE_PATH}")
    if(_cann_root)
      if(SYSTEM_PREFIX)
        list(APPEND _tensor_api_candidates "${_cann_root}/${SYSTEM_PREFIX}/asc")
      endif()
      list(APPEND _tensor_api_candidates "${_cann_root}/asc")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _tensor_api_candidates)

  set(_tensor_api_diagnostics)
  foreach(_candidate IN LISTS _tensor_api_candidates)
    set(_missing_parts)
    foreach(_part IN ITEMS impl/tensor_api include/tensor_api impl/c_api include/c_api)
      file(GLOB_RECURSE _headers LIST_DIRECTORIES false
           "${_candidate}/${_part}/*.h" "${_candidate}/${_part}/*.hpp")
      if(NOT _headers)
        list(APPEND _missing_parts "${_part}")
      endif()
    endforeach()
    if(NOT _missing_parts)
      get_filename_component(_selected "${_candidate}" REALPATH)
      set(TENSOR_API "${_selected}" PARENT_SCOPE)
      message(STATUS "Tensor API headers: ${_selected}")
      return()
    endif()
    list(JOIN _missing_parts ", " _missing_text)
    string(APPEND _tensor_api_diagnostics "\n  ${_candidate}: missing headers in ${_missing_text}")
  endforeach()

  message(FATAL_ERROR
    "No complete Tensor API headers found in the source checkout or selected CANN installation."
    "${_tensor_api_diagnostics}\n"
    "Source the installed CANN toolkit environment and verify its asc header directories. "
    "No empty directories or mixed-version header trees will be substituted.")
endfunction()
