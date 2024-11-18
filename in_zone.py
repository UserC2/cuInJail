def inZone(zx1, zx2, zy1, zy2, bx1, bx2, by1, by2):
    if bx1 < zx1 or bx2 > zx2 or zy1 > by1 or by2 > zy2:
        return False
    return True