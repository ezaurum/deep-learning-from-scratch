class Man:
	def __init__(self, name):
		self.name = name
		print("Initialized")
	def hello(self):
		print("Hello " + self.name + "!")
	def goodby(self):
		print("Good-bye" +self.name)

m = Man("SK")
m.hello()
m.goodby()
