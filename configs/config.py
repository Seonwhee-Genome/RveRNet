import os
import yaml


class Config(object):

    def __init__(self, config_path):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_') # value가 dictionary -> Recursive
                else:
                    self.__setattr__(f'{prefix}{k}', v)

        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.safe_load(file)
        
        __dict2attr(config_dict)        
        self.learning_rate = float(self.learning_rate)
        
    
    def export_to_txt(self, filename):
        with open(filename, 'w') as file:
            file.write(self.__repr__())
        
        
    def __repr__(self):
        str = 'Config(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str
        
        

