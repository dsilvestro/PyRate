#!/usr/bin/env python

from distutils.core import setup, Extension

module1 = Extension('_FastPyRateC',
										include_dirs = ['./'],
                    sources = ['FastPyRateC.cpp', 'FastPyRateC_wrap.cxx'])

setup (name = 'FastPyRateC',
			 author='Xavier Meyer',
		   author_email='xav.meyer@gmail.com',
			 url='https://github.com/dsilvestro/PyRate',
       version = '1.0',
       description = 'This is a package with the main function of PyRate optimzied and implemented in C++.',
       ext_modules = [module1])
