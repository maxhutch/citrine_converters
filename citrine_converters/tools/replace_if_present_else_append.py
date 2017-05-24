def replace_if_present_else_append(objlist, obj, cmp=lambda a,b: a==b):
    """
    Add an object to a list of objects, if that obj does
    not already exist. If it does exist (`cmp(A, B) == True`),
    then replace the property in the property_list. The names
    are compared in a case-insensitive way.

    Input
    =====
    :objlist, list: list of objects.
    :obj, object: object to Add

    Options
    =======
    :cmp, (bool) cmp (A, B): compares A to B. If True, replace.
        If False, append.

    Output
    ======
    None. List is modified in place.
    """
    olist = objlist
    for i in range(len(olist)):
        if cmp(olist[i], obj):
            olist[i] = obj
            return
    # if we get here, then the property was not found. Append.
    olist.append(obj)
