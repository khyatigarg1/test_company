from fuzzywuzzy import fuzz

name='Pepsi - New York'
alias='New York LIfe'
print(fuzz.partial_ratio(name, alias))

name='Pepsi - New York'
alias='PepsiCo Inc.'
print(fuzz.partial_ratio(name, alias))