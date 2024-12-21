from setuptools import find_packages,setup
from typing import List

def install_req(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as libs:
        requirements=libs.readlines()
        requirements=[i.replace("\n", "") for i in requirements]
        print(f"Installing the following dependencies: {requirements}") 
    return requirements

setup(
    name='ml-project',
    version='1',
    author="Shaik Mohammad Huzaifa",
    author_email="Mohammadhuzaifa342a@gmail.com",
    packages=find_packages(),
    install_requires=install_req('requirement.txt')
    )

