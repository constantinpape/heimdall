from .sources import Source


# source wrappers:
# TODO
# - resize on the fly
# - apply affines on the fly
# - data caching

class SourceWrapper(Source):
    pass


class ResizeWrapper(SourceWrapper):
    pass


class AffineWrapper(SourceWrapper):
    pass


class CacheWrapper(SourceWrapper):
    pass
