
_gap = 20
_debug = False
_state = True
def Logger_setup(gap = 20, debug = False, state=True, print_class = False):

    global _gap
    _gap= gap
    global _debug
    _debug = debug
    global _state
    _state = state

def debug(*outs, flush=True):
    if _debug:
        print(*outs, flush=flush)

def state(*outs, flush=True):
    if _state:
        print(*outs, flush=flush)


def results(*outs):
    res = ''
    for i in outs:
        res += "{0:<{gap}}".format(i, gap=_gap)
    print(res, flush=True)

