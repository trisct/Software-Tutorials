class Animal:
    cls_name = 'animal_cls'
    
    def __init__(self):
        self.ins_name = 'animal_ins'
        
    def animal_ins_method(self):
        print(self.ins_name)
    
    @classmethod
    def animal_cls_method(cls):
        print(cls.cls_name)

class Dog(Animal):
    """
    Inherites the Animal class.
    """
    def __init__(self, name):
        self.name = name
    
    def animal_cls_method(self):
        print(f'I am overring the "animal_cls_method" of the base class. You think I\'d be a class method. But I am an instance method. My name is {self.name}')

a = Dog('bob')
a.animal_cls_method()

# This tells us that the methods of the derived class can override that of the base class
