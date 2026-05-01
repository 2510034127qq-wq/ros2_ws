from setuptools import find_packages, setup

package_name = 'thermal_field_reconstructor'

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
    description='Thermal field reconstruction. Image->ThermalField+GetFieldInfo. Route A.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reconstructor_node = thermal_field_reconstructor.reconstructor_node:main',
            'thermal_mapper_node = thermal_field_reconstructor.thermal_mapper_node:main',
        ],
    },
)
