from setuptools import find_packages, setup

package_name = 'signal_preprocessor'

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
    description='Thermal image preprocessing. Kalman/MA/ExpSmooth. Route A. DOI:10.1109/JSEN.2020.2984234',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'preprocessor_node = signal_preprocessor.preprocessor_node:main',
        ],
    },
)
