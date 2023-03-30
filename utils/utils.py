

def color(r="38", g="05", b="222"):
    def my_docorator(func):
        def wrapper(self=None):
            print(f"\033[01;{r};{g};{b}m", end="")
            func() if not self else func(self)
            print("\033[0m")
        return wrapper
    return my_docorator


def coloring(data, r="38", g="05", b="222"):
    color(r, g, b)(lambda: print(data, end=""))()