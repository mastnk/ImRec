#!/usr/bin/env python
# -*- coding: utf-8 -*-

print( 'Enter your ID (e.g. 22M12345): ', end='' )
txt = input().strip()
print()

ascii = [ ord(t) for t in txt ]
s = sum(ascii) % 256
print( 'Your number:', s )
