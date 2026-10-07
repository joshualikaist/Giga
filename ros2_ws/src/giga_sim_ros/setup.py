from glob import glob

from setuptools import find_packages, setup

package_name = 'giga_sim_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joshualikaist',
    maintainer_email='joshualiuniv@gmail.com',
    description='MuJoCo simulator as a ROS2 node (fake robot hardware) for the Giga biped',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # `ros2 run giga_sim_ros sim_node` → giga_sim_ros/sim_node.py 의 main()
            'sim_node = giga_sim_ros.sim_node:main',
            'demo_controller = giga_sim_ros.demo_controller:main',
        ],
    },
)
