# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:11:48 2018
updated on  02/08/2020
@author: David R. Quiroga-Martinez
"""
from export_free_listening import export_free_listening as efl

subjects = list(range(21,35))
blocks = ['piazzolla','hungarian']

for sub in subjects:
      subject = str(sub)
      for block in blocks:
          try:
              efl(subject,block)
          except Exception as e:
              print(e)
              continue
