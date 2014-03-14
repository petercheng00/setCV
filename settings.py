DEBUG = True
DEBUGCONTOURS = False
DEBUGFILL = False
DEBUGSHAPES = False
DEBUGCOLORS = False

color_similarity_threshold = 0.01
shape_similarity_threshold = 8000

fill_empty_half_threshold = 0.4
fill_half_full_threshold = 0.8

def getFillAmount(fillPct):
    if fillPct < fill_empty_half_threshold:
        return 'empty'
    elif fillPct < fill_half_full_threshold:
        return 'half'
    else:
        return 'full'
