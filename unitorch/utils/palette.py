# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

default = {c: (c // 64 * 32, c // 8 % 8 * 32, c % 8 * 32) for c in range(512)}

# fmt: off
_PASCAL = [
    #  class name                id      train          color
    (  'background'            ,  0 ,        0 , (   0,   0,   0) ),
    (  'aeroplane'             ,  1 ,        1 , ( 128,   0,   0) ),
    (  'bicycle'               ,  2 ,        2 , (   0, 128,   0) ),
    (  'bird'                  ,  3 ,        3 , ( 128, 128,   0) ),
    (  'boat'                  ,  4 ,        4 , (   0,   0, 128) ),
    (  'bottle'                ,  5 ,        5 , ( 128,   0, 128) ),
    (  'bus'                   ,  6 ,        6 , (   0, 128, 128) ),
    (  'car'                   ,  7 ,        7 , ( 128, 128, 128) ),
    (  'cat'                   ,  8 ,        8 , (  64,   0,   0) ),
    (  'chair'                 ,  9 ,        9 , ( 192,   0,   0) ),
    (  'cow'                   , 10 ,       10 , (  64, 128,   0) ),
    (  'diningtable'           , 11 ,       11 , ( 192, 128,   0) ),
    (  'dog'                   , 12 ,       12 , (  64,   0, 128) ),
    (  'horse'                 , 13 ,       13 , ( 192,   0, 128) ),
    (  'motorbike'             , 14 ,       14 , (  64, 128, 128) ),
    (  'person'                , 15 ,       15 , ( 192, 128, 128) ),
    (  'pottedplant'           , 16 ,       16 , (   0,  64,   0) ),
    (  'sheep'                 , 17 ,       17 , ( 128,  64,   0) ),
    (  'sofa'                  , 18 ,       18 , (   0, 192,   0) ),
    (  'train'                 , 19 ,       19 , ( 128, 192,   0) ),
    (  'tvmonitor'             , 20 ,       20 , (   0,  64, 128) ),
    (  'borderingregion'       , 255,       21 , ( 224, 224, 192) ),
]

# fmt: on

pascal = {p[2]: p[3] for p in _PASCAL}

palette = {
    "default": default,
    "pascal": pascal,
}


def get(name):
    assert name in palette
    return palette.get(name)
