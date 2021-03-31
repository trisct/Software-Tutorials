# Python Class Tutorials

Below is a python class definition. Some concepts are annotated in the comments.

### Basic Class Attributes

```python
class Dog:
    kind = 'canine'         	# class variable
    all_names = []				# class variable
    def __init__(self, name):
        self.name = name    			# instance variable
        self.all_names.append(name)		# accessing class variable
        
a = Dog('dog_a')
b = Dog('dog_b')

print(a.name)				# accessing instance variable
print(b.name)				# accessing instance variable
print(a.all_names)			# accessing class variable
print(Dog.all_names)		# accessing class variable
print(Dog.name)				# CANNOT access instance variable from a class name

print(a.__class__)			# 'a' is of class '__main__.Dog'
print(Dog.__class__)		# but 'Dog' is of class 'type'
```

### Method Objects

There are three types of method objects:

- instance method
- class method
- abstract method

```python
class Dog:
    category = 'dog'
    
    @classmethod
    def tell_category(cls):					# a class method takes the class object as its first implicit argument
        print(f'I am a {cls.category}')
        
    def tell_name(self, name):				# an instance method takes the instance object as its first implicit argument
        self.name = name
        print(f'My name is {self.name}')
    
    @staticmethod
    def say_something(something):			# uses no implicit arguments
        print(something)
        
a = Dog()
a.tell_category()
a.tell_name('bob')

Dog.tell_category()							# Calling like this is still a class method object
Dog.tell_name(a, 'bob')						# Calling like this is a function object
```

Method definitions in classes override as usual.

```python
class Dog:
    @classmethod
    def m(cls):
        print(cls)
    
    def m(self):							# this overrides the former class method
        print(self)

a = Dog()
a.m()
```

### Class Inheritance

#### Simple Inheritance

```python
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
```

#### Overriding

```python
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
```

#### Use pf `super`

```python
from icecream import ic

class A(object):
    cls_name = 'base_class_A'

class B(A):
    cls_name = 'derived_class_B'

class C(B):
    cls_name = 'second_order_derived_class_C'

class D(A):
    cls_name = 'derived_class_D'
   
ic(A.__mro__)
ic(B.__mro__)
ic(C.__mro__)
ic(D.__mro__)

ic(super(B, B).cls_name)
ic(super(B).cls_name)
ic(super(C, C).cls_name)
ic(super(D, D).cls_name)
ic(super(B, D).cls_name)
ic(super(B, C).cls_name)
```

