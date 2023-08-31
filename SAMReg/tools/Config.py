import json
from logging.config import dictConfig

class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(object):
    def __init__(self, path: str=""):
        if path == "":
            self.data = Dict()
        else:
            self.data = self.load_json(path)

    def save_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=0)
            f.close()

    def __load__(self, data):
        if type(data) is dict:
            return self.load_dict(data)
        else:
            return data

    def load_dict(self, data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = self.__load__(value)
        return result

    def load_json(self, path: str):
        with open(path, "r") as f:
            result = self.__load__(json.loads(f.read()))
        return result
    
    def __missing__(self, key):
        raise ValueError('Could not find key = ' + str( key ) )
    
    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = Dict()
            return self.data[key]
        else:
            return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value