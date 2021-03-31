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
