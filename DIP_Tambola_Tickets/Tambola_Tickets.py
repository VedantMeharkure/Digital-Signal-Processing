#DATE: 10/08/2026
#Author: Vedant Meharkure
import random
ROWS = 3
COLS = 9
def generate_pattern():
    """
    Generate the empty/filled pattern of a Tambola ticket.

    1 = number will be present
    0 = empty cell
    """
    while True:
        pattern = [[0] * COLS for _ in range(ROWS)]
        # Each row gets exactly 5 columns
        for r in range(ROWS):
            selected_columns = random.sample(range(COLS), 5)
            for c in selected_columns:
                pattern[r][c] = 1
        # Check that every column has at least one number
        column_count = [
            sum(pattern[r][c] for r in range(ROWS))
            for c in range(COLS)
        ]
        if all(count >= 1 for count in column_count):
            return pattern

def get_column_range(column):
    """
    Return the number range for a column.
    """
    if column == 0:
        return 1, 9
    elif column == 8:
        return 80, 90
    else:
        start = column * 10
        end = start + 9
        return start, end


def fill_numbers(pattern):
    """
    Fill the pattern with valid Tambola numbers.
    """
    ticket = [[None] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        start, end = get_column_range(c)
        # Numbers available for this column
        numbers = list(range(start, end + 1))
        random.shuffle(numbers)
        # Find rows where this column needs a number
        rows = [
            r for r in range(ROWS)
            if pattern[r][c] == 1
        ]
        # Select required number of values
        selected = numbers[:len(rows)]
        # Numbers must be sorted vertically
        selected.sort()
        for r, number in zip(rows, selected):
            ticket[r][c] = number
    return ticket


def generate_ticket():
    """
    Generate a complete Tambola ticket.
    """
    pattern = generate_pattern()
    ticket = fill_numbers(pattern)
    return ticket

def print_ticket(ticket):
    """
    Print ticket in a proper table format.
    """
    print()
    print("+" + "-------+" * 9)
    for row in ticket:
        print("|", end="")
        for number in row:
            if number is None:
                print("       |", end="")
            else:
                print(f"{number:^7}|", end="")
        print()
        print("+" + "-------+" * 9)
    print()

# --------------------------------------------------
# MAIN
# --------------------------------------------------

ticket = generate_ticket()
print("              TAMBOLA TICKET")
print_ticket(ticket)