import setuptools 



setuptools.setup(  
    name='unrolled',
    version='0.1',
    description='Code that goes with EUSIPCO paper',
    url='y-mhiri.github.io',
    author='y-mhiri',
    install_requires=['numpy', 
                     'scipy', 
                     'scikit-image',
                     'ducc0',
                     'zarr',
                     'click',
                     'astropy',
                     'ducc0',
                     'torch',
                     'torchvision',
                     'matplotlib',
                     'numcodecs',
                     'python-casacore',
                     'flask-caching',
                     'omegaconf',
                     'dash'],
    entry_points={
            'console_scripts': [
                'generate_dataset = unrolled.astro.astro:generate_dataset', 
                'train_model = unrolled.deep.train:train',
            ]
    },
    author_email='yassine.mhiri@outlook.fr',
    packages=setuptools.find_packages(),
    zip_safe=False
        )