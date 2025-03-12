import numpy as np


# Linear domain
def linear(size, boundary):

    # Set the number of cell pairs
    pairs = size
    if boundary == "dirichlet":
        pairs = size - 1

    # Generate the cell pairs
    output = [] 
    for i in range(pairs):
        output.append([i, (i+1) % size])
        output.append([(i+1) % size, i])

    return np.array(output), size


# TEST CASES
# print(linear(2, "dirichlet"))
# print(linear(10, "dirichlet"))
# print(linear(3, "periodic"))
# print(linear(10, "periodic"))



# Hexagonal domains are paralellograms, indexed from left to right, top to bottom 

# EXAMPLE:
#
# 00 01 02 03 04 
#  05 06 07 08 09
#   10 11 12 13 14
#    15 16 17 18 19
#     20 21 22 23 24

# Hexagonal domain with zero-flux boundary
def hexagonal_dirichlet(height, width):

    # Generate the parallelogram row by row
    output = []
    for h in range(height):

        # Add all neighbours of each hexagon
        for w in range(width):

            # Get the index of the current cell
            i = (h * width) + w

            # Top left neighbour
            if h > 0: output.append([i, i - width])

            # Top right neighbour
            if h > 0 and w < width - 1: output.append([i, i - width + 1])

            # Left neighbour
            if w > 0: output.append([i, i - 1])

            # Right neighbour
            if w < width - 1: output.append([i, i + 1])

            # Bottom left neighbour
            if h < height - 1 and w > 0: output.append([i, i + width - 1])

            # Bottom right neighbour
            if h < height - 1: output.append([i, i + width])

    return np.array(output), (height * width)


# Hexagonal domain with periodic boundary
def hexagonal_periodic(height, width):

    # Generate the parallelogram row by row
    output = []
    for h in range(height):

        # Add all neighbours of each hexagon
        for w in range(width):

            # Get the row of this cell, and the row above/below
            rt = ((h - 1) % height) * width
            rm = (h * width)
            rb = ((h + 1) % height) * width

            # Top left neighbour
            output.append([rm + w, rt + w])

            # Top right neighbour
            output.append([rm + w, rt + ((w + 1) % width)])

            # Left neighbour
            output.append([rm + w, rm + ((w - 1) % width)])

            # Right neighbour
            output.append([rm + w, rm + ((w + 1) % width)])

            # Bottom left neighbour
            output.append([rm + w, rb + ((w - 1) % width)])

            # Bottom right neighbour
            output.append([rm + w, rb + w])

    return np.array(output), (height * width)


# Master function to call the correct helper
def hexagonal(height, width, boundary):
    if boundary == "dirichlet":
        return hexagonal_dirichlet(height, width)
    if boundary == "periodic":
        return hexagonal_periodic(height, width)


# TEST CASES
# print(hexagonal(1, 5, "dirichlet"))
# print(hexagonal(5, 1, "dirichlet"))
# print(hexagonal(5, 5, "dirichlet"))
# print(hexagonal(1, 5, "periodic"))
# print(hexagonal(5, 1, "periodic"))
# print(hexagonal(5, 5, "periodic"))
