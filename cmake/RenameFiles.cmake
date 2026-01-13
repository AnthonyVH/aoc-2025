function(RenameFiles FILE_DIR NEEDLE REPLACEMENT)
    message(DEBUG "Renaming files in '${FILE_DIR}': replacing '${NEEDLE}' with '${REPLACEMENT}'")
    file(GLOB FILES "${FILE_DIR}/*${NEEDLE}*")
    message(DEBUG "  Found ${FILES}")
    foreach(FILE ${FILES})
        get_filename_component(FILENAME "${FILE}" NAME)
        string(REPLACE "${NEEDLE}" "${REPLACEMENT}" NEW_FILENAME "${FILENAME}")
        message(DEBUG "    Renaming file: ${FILE} to ${NEW_FILENAME}")
        file(RENAME "${FILE}" "${FILE_DIR}/${NEW_FILENAME}")
    endforeach()
endfunction()

function(GetRenameFilesPath RESULT_VAR)
    # Return the path to this file, so it can be invoked via 'cmake -P'.
    set(${RESULT_VAR} "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" PARENT_SCOPE)
endfunction()

# This triggers when run via 'cmake -P'.
if(CMAKE_SCRIPT_MODE_FILE)
    RenameFiles(${FILE_DIR} ${NEEDLE} ${REPLACEMENT})
endif()
