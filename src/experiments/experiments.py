import os
import abc

class NoAttributesException(Exception):

    def _message_one_attribute(self, arg):
        return "attribute: '{}' has not been provided".format(arg)
    
    def _message_many_attributes(self, *args):
        message = "attributes: "  
        message += ", ".join(map(lambda x: "'{}'".format(x), args))
        message += " have not bee provided"
        return message

    def __init__(self, obj, *args):
        message = "In class '{}': ".format(type(obj).__name__)
        if len(args) == 0:
            message = "<Incorrect NoAttributesException usage>"
        elif len(args) == 1:
            message += self._message_one_attribute(args[0])
        else:
            message += self._message_many_attributes(*args)
        super().__init__(self, message)

def check_presence(*args):
    def decorator(function):
        def wrapper(self, state, *_args, **_kwargs):
            self._check_attributes_presence(state, *args)
            return function(self, state, *_args, **_kwargs)
        return wrapper
    return decorator

class RunnableItem:

    def _check_attributes_presence(self, state, *args):
        missed = []
        for arg in args:
            try:
                getattr(state, arg)
            except AttributeError:
                missed.append(arg)
        if len(missed) > 0:
            raise NoAttributesException(self, *missed)

    def __init__(self):
        pass
    

    def _preprocessing(self, state):
        self_class = self.__class__
        try:
            mro = self.__class__.__mro__[:-1]
            r_mro = tuple(reversed(mro))
            for _cls in r_mro:
                if 'preprocessing' in _cls.__dict__.keys():
                    self.__class__ = _cls
                    self.preprocessing(state)
        except:
            self.__class__ = self_class
            raise
        self.__class__ = self_class

    
    def _postprocessing(self, state):
        self_class = self.__class__
        try:
            mro = self.__class__.__mro__[:-1]
            for _cls in mro:
                if 'postprocessing' in  _cls.__dict__.keys():
                    self.__class__ = _cls
                    self.postprocessing(state)
        except:
            self.__class__ = self_class
            raise
        self.__class__ = self_class

    def preprocessing(self, state):
        pass

    def postprocessing(self, state):
        pass

    def run(self, state):
        state.attach_level() # level for local variables (created during preprocessing)

        try:
            self._preprocessing(state)
        except:
            state.detach_level()
            raise
        
        state.attach_level() # level for internal run results

        try:
            self.internal_run(state)
            self._postprocessing(state) # postprocessing works with both result and local arguments levels
        except:
            state.detach_level()
            state.detach_level()
            raise
    
        res_level_dict = state.get_level_dict() # get results level dict
        state.detach_level() # detach results level
        state.detach_level() # detach arguments level
        state.attach_current(**res_level_dict) # add results level arguments to the state
    
    @abc.abstractmethod
    def internal_run(self, state):
        pass