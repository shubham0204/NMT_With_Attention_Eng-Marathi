
import json


def write_config( filepath , train_config ):
    with open( filepath , 'w' ) as fp:
        json.dump( train_config , fp )

def read_config( filepath ):
    with open( filepath , 'r' ) as fp:
        config = json.load( fp )
    return config
