from setuptools import find_packages, setup

package_name = 'thermal_motion_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thermal_robot',
    maintainer_email='thermal_robot@example.com',
    description='Thermal motion controller. max_gradient+PID. Route A. DOI:10.1109/ROBOT.2006.1642286',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = thermal_motion_controller.controller_node:main',
            'source_tracker_node = thermal_motion_controller.source_tracker_node:main',
        ],
    },
)
