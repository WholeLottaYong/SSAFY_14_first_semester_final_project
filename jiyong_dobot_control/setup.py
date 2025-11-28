from setuptools import find_packages, setup
import os # 경로 생성기
from glob import glob # 특정 파일 전부 가져오기

package_name = 'jiyong_dobot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # launch 폴더 안의 모든 .launch.py 파일을 install/share/패키지명/launch 로 복사
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        
        # maps 폴더 안의 모든 파일(.yaml, .pgm 등)을 install/share/패키지명/maps 로 복사
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ssafy',
    maintainer_email='ssafy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cube_detector = jiyong_dobot_control.cube_detector:main',
            'calc_matrix = jiyong_dobot_control.calc_matrix:main',
            'scanner_only = jiyong_dobot_control.scanner_only:main',
            'scanner_average = jiyong_dobot_control.scanner_average:main',
            'dobot_test = jiyong_dobot_control.dobot_test:main',
            'dobot_test2 = jiyong_dobot_control.dobot_test2:main',
            'dobot_test3 = jiyong_dobot_control.dobot_test3:main',
        ],
    },
)
