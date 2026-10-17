import os
from glob import glob
from setuptools import setup

package_name = 'postman_solution'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Participant Team',
    maintainer_email='team@postbot.org',
    description='Participant solution package for Sim2Real Robotics Hackathon (Simulation Round)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'solution_node = postman_solution.solution_node:main',
            'tier1_node    = postman_solution.tier1_node:main',
            'tier2_node    = postman_solution.tier2_node:main',
            'tier3_node    = postman_solution.tier3_node:main',
        ],
    },
)
