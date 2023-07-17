def lambda_is_ok_upper_bound_int(lambda_is_ok, left, right):
    while left < right:
        middle = (left + right) // 2
        if lambda_is_ok(middle):
            left = middle + 1
        else:
            right = middle

    return left
