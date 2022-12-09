def deEncoder(x) :
    denc = []

    for i in x :
        if i == [1,0,0,0,0,0,0,0,0,0] :
            denc.append(0)
        elif i == [0,1,0,0,0,0,0,0,0,0] :
            denc.append(1)
        elif i == [0,0,1,0,0,0,0,0,0,0] :
            denc.append(2)
        elif i == [0,0,0,1,0,0,0,0,0,0] :
            denc.append(3)
        elif i == [0,0,0,0,1,0,0,0,0,0] :
            denc.append(4)
        elif i == [0,0,0,0,0,1,0,0,0,0] :
            denc.append(5)
        elif i == [0,0,0,0,0,0,1,0,0,0] :
            denc.append(6)
        elif i == [0,0,0,0,0,0,0,1,0,0] :
            denc.append(7)
        elif i == [0,0,0,0,0,0,0,0,1,0] :
            denc.append(8)
        elif i == [0,0,0,0,0,0,0,0,0,1] :
            denc.append(9)
    return denc