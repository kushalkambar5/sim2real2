#!/usr/bin/env python3
"""
solution_node.py
Unified entry point for Sim2Real Competition.
Dispatches to tier1_node, tier2_node, or tier3_node based on the specified tier argument or parameter.
"""
import sys
from postman_solution import tier1_node, tier2_node, tier3_node


def main():
    tier = 1
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith('tier:='):
            try:
                tier = int(arg.split(':=')[1])
            except ValueError:
                pass
        elif arg.startswith('--tier='):
            try:
                tier = int(arg.split('=')[1])
            except ValueError:
                pass
        elif arg in ('--tier', '-t') and i + 1 < len(args):
            try:
                tier = int(args[i + 1])
            except ValueError:
                pass
        elif arg in ('1', '2', '3'):
            tier = int(arg)

    if tier == 2:
        tier2_node.main()
    elif tier == 3:
        tier3_node.main()
    else:
        tier1_node.main()


if __name__ == '__main__':
    main()
