# class
class Awesome:

    # some method
    def greetings(self):
        print("Hello World!")

    # the del method
    def __del__(self):
        print("Hello from the __del__ method.")

# object of the class
obj = Awesome()

# calling class method
obj.greetings()

print(obj)
