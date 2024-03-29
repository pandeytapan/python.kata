- The `defaultdict` calls the `__default_factory__` only for the `__getitem__`.
- The underlying thing that actually gets called is the `__missing__`.
- Derive a class from the `dict` and provide it with the `__missing__` now instead of the `KeyError` we will see the `__missing__` getting called for the `__getitem__`.
- When using `[]` for the dictionary it uses the `getitem` to retrieve the values. However, `get` is used exclusively to retrieve the values.
- In such cases the `get` never makes the use of the `[]`.
- Whenever we have a missing key the `__getitem__` will call the `__missing__`. This gives us the opportunity to handle the missing key and provide one
- The `collections.OrderedDict` is useful in situations when we want to mainitain the order of insertion as well the comparison. Consider the example below

# Regular dictionaries in Python

dict1 = {'apple': 3, 'pear': 1, 'orange': 4}
dict2 = {'pear': 1, 'apple': 3, 'orange': 4}

# Order doesn't matter in the regular Python dictionaries

dict1 == dict2 # This is true

ord_dict1 = OrderedDict([('apple': 3), ('pear': 1), ('orange': 4)])
ord_dict2 = OrderedDict([('pear': 1), ('apple': 3), ('orange': 4)])

# Order matters in the ordered dictionaries
ord_dict1 == ord_dict2 # This will return the false
- What can be added as a key to the dictionary or what can be added to a set is known as hashable.
- Hashable elements must: 
- Be immutable i.e. their value cannot be changed after they're crea
- Must implement the hash function.
- Must implement the `__eq__`
- A container is hashable if all its elements are hashable.
- A set in literal format is written as {1, 2, 3, 4}
- An empty set is written as `set()` and an empty dictionary is written as `{}`.
- Like list comprehension we also have set comprehension in Python
- There is no special symbol to create a frozenset(), we must creates it using the constructor `frozenset()`
