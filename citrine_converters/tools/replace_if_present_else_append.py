def replace_if_present_else_append(
        objlist, obj, cmp=lambda a,b: a==b, rename=None):
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
    :cmp, (bool) cmp (A, B): compares A to B. If True, then the
        objects are the same and B should replace A. If False,
        then B should be appended to `objlist`.
    :rename, (type(A)) rename(A): Change A (the object already
        present in the list) such that if `cmp(A, B) --> True`,
        then `cmp(rename(A), B) --> False. This property of
        `rename` is checked, and if both return True (causing an
        infinite loop), an exception (ValueError) is raised. This
        function returns another A-like object (or a reference to
        A itself).

    Output
    ======
    None. List is modified in place.
    """
    for i in range(len(objlist)):
        # was a matching object found in the list?
        if cmp(objlist[i], obj):
            # if so, should the old object be renamed?
            if rename is not None:
                newA = rename(objlist[i])
                # is the renamed object distinct from the object
                # (`obj`) that is to be added to the list?
                if cmp(newA, obj):
                    msg = '`rename` does not make {} unique.'.format(
                        str(objlist[i])[:32])
                    raise ValueError(msg)
                # now that we have newA, replace the original
                # object in the list with `obj`...
                objlist[i] = obj
                #... and replace_if_present_else_append newA.
                replace_if_present_else_append(
                    objlist, newA, cmp=cmp, rename=rename)
            # if the existing object should not be renamed,
            # simply replace.
            else:
                objlist[i] = obj
            # short circuit to exit the for loop and the function.
            return
    # if we get here, then the property was not found. Append.
    objlist.append(obj)
