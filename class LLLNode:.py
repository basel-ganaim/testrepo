def somePairs(limit):
    i = 0
    while i < limit:
        for j in range(i):
            yield (i, j, i, j)
        i += 1

# Example usage with a limit
pairs_generator = somePairs(10000000000)

# Print all generated pairs
for pair in pairs_generator:
    print(pair)
