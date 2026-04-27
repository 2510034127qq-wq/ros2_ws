from setuptools import find_packages, setup

package_name = 'thermal_gradient_processor'

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
    description='Thermal gradient processor. Sobel/central-diff. Route A. DOI:10.1016/j.robot.2020.103687',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gradient_node = thermal_gradient_processor.gradient_node:main',
        ],
    },
)
