import pprint

pp = pprint.PrettyPrinter(indent=4, depth=6, width=160)

def to_string(obj):
    return pp.pformat(obj.__dict__)