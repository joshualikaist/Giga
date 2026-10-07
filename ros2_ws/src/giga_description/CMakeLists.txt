cmake_minimum_required(VERSION 3.8)
project(giga_description)

find_package(ament_cmake REQUIRED)

# 코드는 없고 파일만 설치한다 → install/giga_description/share/giga_description/{urdf,launch,rviz}
# (--symlink-install로 빌드하면 원본 파일을 가리키는 링크라서, URDF를 고쳐도 다시 빌드할 필요 없음)
install(DIRECTORY urdf launch rviz
  DESTINATION share/${PROJECT_NAME})

ament_package()
