color_similarity_threshold = 0
shape_similarity_threshold = 2000

fill_empty_half_threshold = 0.4
fill_half_full_threshold = 0.8

def getFillAmount(fillPct):
    if fillPct < fill_empty_half_threshold:
        return 'empty'
    elif fillPct < fill_half_full_threshold:
        return 'half'
    else:
        return 'full'
