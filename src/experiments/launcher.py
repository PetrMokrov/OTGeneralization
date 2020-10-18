from multipledispatch import dispatch
import warnings

from .state import State
from .experiments import RunnableItem

class _ExperimentLocalizer:

    def __init__(self, launcher, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args
        self.launcher = launcher
    
    def __enter__(self):
        self.launcher.state.attach_level(**self.kwargs)
        return self.launcher
    
    def __exit__(self, *exc):
        level_dict = self.launcher.state.get_level_dict()
        self.launcher.state.detach_level()
        missed_args = []
        attach_dict = {}
        for arg in self.args:
            if not arg in level_dict.keys():
                missed_args.append(arg)       
            else:
                attach_dict[arg] = level_dict[arg]
        
        if len(attach_dict) > 0:
            self.launcher.state.attach_current(**attach_dict)
        if len(missed_args) > 0:
            str_missed = ", ".join(map(str, missed_args))
            warnings.warn(
                "Arguments ({}) to be passed up-level," 
                "but not presented in the state".format(str_missed))


class ExperimentLauncher:

    def __init__(self, create_level=False):
        self.state = State()
        if create_level:
            self.state.attach_level()
    
    def localize(self, *args, **kwargs):
        return _ExperimentLocalizer(self, *args, **kwargs)
    
    def specify(self, **kwargs):
        self.state.attach_current(**kwargs)
    
    def delete(self, *args):
        self.state.detach_current(*args)
    
    def _launch(self, runnable):
        runnable.run(self.state)
    
    @dispatch(RunnableItem)
    def launch(self, runnable, *args, **kwargs):
        self.specify(**kwargs)
        self._launch(runnable)
        self.delete(*args)
    
    @dispatch(RunnableItem)
    def launch_safe(self, runnable, **kwargs):
        with self.localize(**kwargs):
            self._launch(runnable)
