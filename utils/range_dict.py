class RangeKeyDict:
    def __init__(self, my_dict):
        assert not any(map(lambda x: not isinstance(x, tuple) or len(x) != 2 or x[0] > x[1], my_dict))

        def lte(bound):
            return lambda x: bound <= x

        def gt(bound):
            return lambda x: x < bound

        # generate the inner dict with tuple key like (lambda x: 0 <= x, lambda x: x < 100)
        self._my_dict = {(lte(k[0]), gt(k[1])): v for k, v in my_dict.items()}

    def __getitem__(self, number):
        from functools import reduce
        _my_dict = self._my_dict
        try:
            result = next((_my_dict[key] for key in _my_dict if list(reduce(lambda s, f: filter(f, s), key, [number]))))
        except StopIteration:
            raise KeyError(number)
        return result

    def get(self, number, default=None):
        try:
            return self.__getitem__(number)
        except KeyError:
            return default
