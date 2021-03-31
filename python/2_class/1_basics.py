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
