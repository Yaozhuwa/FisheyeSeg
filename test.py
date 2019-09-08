class Foo(object):
    def __init__(self):
        self.name = self.__class__.__name__


    def cout(self):
        print(self.name)



class Bar(Foo):
    pass


b = Bar()
b.cout()