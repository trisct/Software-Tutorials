class Dog:
    @classmethod
    def m(cls):
        print(cls)
    
    def m(self):							# this overrides the former class method
        print(self)

a = Dog()
a.m()
