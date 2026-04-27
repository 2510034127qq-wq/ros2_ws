cmake_minimum_required(VERSION 3.8)
project(g1_description)

find_package(ament_cmake REQUIRED)

# Install URDF files (original + ros2_control version)
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.urdf" PATTERN "*.xacro"
)

# Install meshes (user must copy *.STL here before colcon build)
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# Install config (ros2_control.yaml etc.)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
