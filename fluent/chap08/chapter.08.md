# Object references, mutability and Recycling

The name is not the object, the name is a separate thing

## Variables are not boxes

Python variables are like references in the java or like the pointers in the C/C++. It basically can be treated as a label attached to the object (instead of being like object itself). This can be seen in list example below where the two variables hold reference to same list instead of separate copies.

```python
>>> a = [1, 2, 3, 4]
>>> b = a
>>> b.append(5)
>>> a
[1, 2, 3, 4, 5]
```

We can see that appending a new value using `b` is clearly visible thru the `a`

**Variables are assigned to objects** instead of common notion **Objects are assigned to the variables**. Here is an example below:

```python
>>> class Gizmo:
...
...     def __init__(self):
...             print(f"Gizmo id: {self}")
...
>>> Gizmo()
Gizmo id: <__main__.Gizmo object at 0x100e2e410>
<__main__.Gizmo object at 0x100e2e410>
>>> x = Gizmo()
Gizmo id: <__main__.Gizmo object at 0x100f00dd0>
>>> y = Gizmo() * 10
Gizmo id: <__main__.Gizmo object at 0x100f00d90>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for *: 'Gizmo' and 'int'
```

In the class above `Gizmo` we're printing the memory location where the object is stored when ever a new object is inititalized. In the example `x = Gizmo()`, first the `Gizmo()` is created and then the reference is stored in the variable `x`.
In the next example `Gizmo() * 10` the `Gizmo` is getting created but since multiplication is not defined so we're getting an error.

## Identity, Equality and Aliases

Every object in Python occupies memory. We've a list here referenced by `charles` and we stored the reference in another variable `lewis`:

```python
>>> charles = {'name': 'Charles L. Dodgson', 'born': 1832}
>>> lewis = charles
>>> lewis
{'name': 'Charles L. Dodgson', 'born': 1832}
>>> charles
{'name': 'Charles L. Dodgson', 'born': 1832}
>>> id(lewis)
4365979200
>>> id(charles)
4365979200
```

When we printed the memory location of the variable, `charles` and the `lewis` we got the same values `4365979200`. 

```python
>>> lewis == charles
True
>>> lewis is charles
True
>>> alex = {'name': 'Charles L. Dodgson', 'born': 1832}
>>> alex == charles
True
>>> alex is not charles
True
>>> alex is charles
False
>>> id(alex)
4367404992
```
