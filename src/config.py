# set this flag to True to get additional configuration options for e.g. anchor ratios
EXTENDED_PY_FASTER_RCNN = False
try:
    import private_config

    if private_config.EXTENDED_PY_FASTER_RCNN != None:
        EXTENDED_PY_FASTER_RCNN=private_config.EXTENDED_PY_FASTER_RCNN

except:
    pass #ignore if not exists

