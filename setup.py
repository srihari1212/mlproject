from setuptools import find_packages, setup

edot = '.e .'

def get_requirements(filepath:str)->List[str]:
    '''
    this function will return the list of packages
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [each.replace('\n', '') for each in requirements]
        if edot in requirements:
            requirements.remove(edot)
    


setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'sri hari',
    author_email = 'sriharivenkatesan10488@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)