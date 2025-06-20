from termcolor import colored


def pc(key, value, color='blue', break_line=False):
    new_line = '\n' if break_line else ''
    print(colored(key, color) + ': {}{}'.format(value, new_line))
