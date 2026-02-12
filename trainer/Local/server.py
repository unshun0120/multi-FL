""" Local Train. """
# Use center server here.
from ..BaseFL.server import Server as Base_Server


class Server(Base_Server):

    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)
        self.algorithm_name = "Local"

    """use the run process of Center, i.e. pass all FL process except metric model"""

    def distribute_model(self):
        """server don't distribute model anymore"""
        pass

