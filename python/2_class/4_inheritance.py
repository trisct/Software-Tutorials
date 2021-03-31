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
    pass

a = Dog()
a.animal_cls_method()       # calling the method in Animal
a.animal_ins_method()       # calling the method in Animal

print(a.cls_name)           # accessing the base class attribute

Dog.cls_name = 'dog_cls'    # NOT modifying the base class attribute, but rather, creates a new attribute for Dog          
a.cls_name = 'dog_clsins'   # NOT modifying the base class attribute, nor the class attribute, but rather, creates a new attribute for the instance
Animal.animal_cls_method()  # NOT modifying the base class attribute
Dog.animal_cls_method()     # calling base class method, but access the attribute 'cls_name' of Dog, not of the base class 
print(a.cls_name)           # accessing the instance attribute
