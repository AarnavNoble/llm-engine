include(Catch)

add_executable(engine_tests
  test_smoke.cpp
)
target_link_libraries(engine_tests PRIVATE engine_core Catch2::Catch2WithMain)
target_compile_definitions(engine_tests PRIVATE
  ENGINE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  ENGINE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/Qwen2.5-0.5B-Instruct")
catch_discover_tests(engine_tests)
