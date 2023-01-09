#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2023/01/01 21:29:25
@Author  :   ronghao
@Version :   1.0
@Contact :   ronghaoli1997@qq.com
@Desc    :   读取配置文件类
'''
import yaml

class CFG:
    def __init__(self) -> None:
        self._dict = {}
        

    def from_config_yaml(self,config_path):
        self._dict = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
        # self._dict['STATUS']['CONFIG'] = config_path
    
    def from_dict(self,config_dict):
        self._dict = config_dict



