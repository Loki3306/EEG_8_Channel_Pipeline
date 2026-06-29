def factorial(n: int) -> int:
    """Calculate the factorial of n."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer.")
        
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    print(factorial(5))
