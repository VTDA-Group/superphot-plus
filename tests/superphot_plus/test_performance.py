import pytest

def fib(n):
    if n <= 1:
        return 1
    return fib(n - 2) + fib(n - 1)

def test_fib_10():
    fib(30)

def test_fib_20():
    fib(40)