from setuptools import find_packages, setup

package_name = 'thermal_sensor_sim'

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
    description='Thermal sensor sim + colorizer nodes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_node     = thermal_sensor_sim.sensor_node:main',
            'colorizer_node  = thermal_sensor_sim.colorizer_node:main',
        ],
    },
)
