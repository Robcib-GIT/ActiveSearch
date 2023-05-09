'''
    Variables globales del sistema, que indican si se está o no en modo TEST
    Jorge F. García-Samartín
    www.gsamartin.es
    29-04-2022
'''

TEST = False   
TEST_LOW = False
VERBOSE = True
TAGS = True             # True para que las etiquetas de COCO se carguen solas (más rápido)
SEARCH_DOORS = False    # True para que se busquen puertas en el mapa
SAVE_POS = True         # True para que se guarde el histórico de posiciones del robot
NAVE = False             # True para que no pueda ir a posiciones fuera del recinto con reja de la nave